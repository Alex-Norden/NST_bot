"""
Multi-style Generative Network for Real-time Transfer
ref https://arxiv.org/abs/1703.06953
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class GramMatrix(nn.Module):
	def __init__(self):
		super(GramMatrix, self).__init__()

	def forward(self, y):
		(b, ch, h, w) = y.size()
		features = y.view(b, ch, w * h)
		features_t = features.transpose(1, 2)
		gram = features.bmm(features_t) / (ch * h * w)
		return gram


class Inspiration(nn.Module):
	"""
	Inspiration Layer (from MSG-Net paper)
	tuning the featuremap with target Gram Matrix
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, C, B=1):
		super(Inspiration, self).__init__()
		# B is equal to 1 or input mini_batch
		self.weight = nn.Parameter(torch.Tensor(1, C, C), requires_grad=True)
		# non-parameter buffer
		self.G = Variable(torch.Tensor(B, C, C), requires_grad=True)
		self.C = C
		self.reset_parameters()

	def reset_parameters(self):
		self.weight.data.uniform_(0.0, 0.02)

	def setTarget(self, target):
		self.G = target

	def forward(self, X):
		# input X is a 3D feature map
		self.P = torch.bmm(self.weight.expand_as(self.G), self.G)
		return torch.bmm(self.P.transpose(1, 2).expand(X.size(0), self.C, self.C),
						X.view(X.size(0), X.size(1), -1)).view_as(X)

	def __repr__(self):
		return "{}(N x {})".format(self.__class__.__name__, self.C)


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super(ConvLayer, self).__init__()
		reflection_padding = int(np.floor(kernel_size / 2))
		self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		out = self.reflection_pad(x)
		return self.conv2d(out)


class UpsampleConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
		super(UpsampleConvLayer, self).__init__()

		if upsample:
			self.upsample_layer = nn.Upsample(scale_factor=upsample)
		self.upsample = upsample

		self.reflection_padding = int(np.floor(kernel_size / 2))
		if self.reflection_padding != 0:
			self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)

		self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

	def forward(self, x):
		if self.upsample:
			x = self.upsample_layer(x)
		if self.reflection_padding != 0:
			x = self.reflection_pad(x)
		return self.conv2d(x)


class Bottleneck(nn.Module):
	""" Pre-activation residual block
	Identity Mapping in Deep Residual Networks
	ref https://arxiv.org/abs/1603.05027
	"""
	def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
		super(Bottleneck, self).__init__()
		self.expansion = 4
		if downsample is not None:
			self.residual_layer = nn.Conv2d(inplanes, planes * self.expansion,
											kernel_size=1, stride=stride)
		self.downsample = downsample

		self.conv_block = nn.Sequential(
				norm_layer(inplanes),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),

				norm_layer(planes),
				nn.ReLU(inplace=True),
				ConvLayer(planes, planes, kernel_size=3, stride=stride),

				norm_layer(planes),
				nn.ReLU(inplace=True),
				nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
			)

	def forward(self, x):
		if self.downsample is not None:
			return self.residual_layer(x) + self.conv_block(x)
		else:
			return x + self.conv_block(x)


class UpBottleneck(nn.Module):
	""" Up-sample residual block (from MSG-Net paper)
	Enables passing identity all the way through the generator
	ref https://arxiv.org/abs/1703.06953
	"""
	def __init__(self, inplanes, planes, stride=2, norm_layer=nn.BatchNorm2d):
		super(UpBottleneck, self).__init__()
		self.expansion = 4
		self.residual_layer = UpsampleConvLayer(inplanes,
												planes * self.expansion,
												kernel_size=1, stride=1,
												upsample=stride)
		self.conv_block = nn.Sequential(
				norm_layer(inplanes),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes, planes, kernel_size=1, stride=1),

				norm_layer(planes),
				nn.ReLU(inplace=True),
				UpsampleConvLayer(planes, planes, kernel_size=3, stride=1, upsample=stride),

				norm_layer(planes),
				nn.ReLU(inplace=True),
				nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1)
			)

	def forward(self, x):
		return self.residual_layer(x) + self.conv_block(x)


class MSGNet(nn.Module):
	"""The MSG-Net"""
	def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6):
		super(MSGNet, self).__init__()
		self.gram = GramMatrix()

		block = Bottleneck
		upblock = UpBottleneck
		expansion = 4

		self.model1 = nn.Sequential(
				ConvLayer(input_nc, 64, kernel_size=7, stride=1),
				norm_layer(64),
				nn.ReLU(inplace=True),
				block(64, 32, 2, 1, norm_layer),
				block(32 * expansion, ngf, 2, 1, norm_layer)
			)

		self.ins = Inspiration(ngf * expansion)

		self.model = nn.Sequential(
				self.model1,
				self.ins,

				*(block(ngf * expansion, ngf, 1, None, norm_layer) for _ in range(n_blocks)),

				upblock(ngf * expansion, 32, 2, norm_layer),
				upblock(32 * expansion, 16, 2, norm_layer),
				norm_layer(16 * expansion),
				nn.ReLU(inplace=True),
				ConvLayer(16 * expansion, output_nc, kernel_size=7, stride=1)
			)

	def setTarget(self, Xs):
		f = self.model1(Xs)
		G = self.gram(f)
		self.ins.setTarget(G)

	def forward(self, input):
		return self.model(input)