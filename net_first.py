"""
A Neural Algorithm of Artistic Style
ref https://arxiv.org/abs/1508.06576
"""
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		# we "detach" the target content from the tree used
		# to dynamically compute the gradient: this is a stated value,
		# not a variable. Otherwise the forward method of the criterion
		# will throw an error.
		self.target = target.detach()
		self.loss = F.mse_loss(self.target, self.target) #to initialize with something

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input


def gram_matrix(input):
	batch_size , h, w, f_map_num = input.size() # batch size(=1)
	# b=number of feature maps
	# (h,w)=dimensions of a feature map (N=h*w)

	features = input.view(batch_size * h, w * f_map_num) # resise F_XL into \hat F_XL

	G = torch.mm(features, features.t()) # compute the gram product

	# we "normalize" the values of the gram matrix
	# by dividing by the number of element in each feature maps.
	return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()
		self.loss = F.mse_loss(self.target, self.target) #to initialize with something

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input


class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		# .view the mean and std to make them [C x 1 x 1] so that they can
		# directly work with image Tensor of shape [B x C x H x W].
		# B is batch size. C is number of channels. H is height and W is width.
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, img):
		# normalize img
		return (img - self.mean) / self.std


class GatysNet(nn.Module):
	"""The Gatys-Net"""
	def __init__(self):
		super(GatysNet, self).__init__()

		self.features = nn.Sequential(OrderedDict([
			("conv_1", nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
			("relu_1", nn.ReLU(inplace=False)),
			("conv_2", nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
			("relu_2", nn.ReLU(inplace=False)),
			("pool_2", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
			("conv_3", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
			("relu_3", nn.ReLU(inplace=False)),
			("conv_4", nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
			("relu_4", nn.ReLU(inplace=False)),
			("pool_4", nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
			("conv_5", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
		]))

	def forward(self, input):
		return self.features(input)


class GatysTransfer(object):
	def __init__(self, cnn):
		super(GatysTransfer, self).__init__()

		self.cnn = cnn
		# normalization module
		self.normalization = Normalization(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.485, 0.456, 0.406]))

		self.content_layers = ("conv_4",)
		self.style_layers = ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")

		self.model = None
		self.step = 0

		self.content_img = None
		self.style_img = None

		self.content_losses = []
		self.style_losses = []

		self.prev_score = None

	def build_model(self, content_img, style_img):
		self.content_img = content_img
		self.style_img = style_img

		# just in order to have an iterable access to list of content/style losses
		self.content_losses.clear()
		self.style_losses.clear()

		cnn = copy.deepcopy(self.cnn)

		# assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
		# to put in modules that are supposed to be activated sequentially
		model = nn.Sequential(self.normalization)

		i = 0 # increment every time we see a conv
		for layer in cnn.children():
			if isinstance(layer, nn.Conv2d):
				i += 1
				name = "conv_{}".format(i)
			elif isinstance(layer, nn.ReLU):
				name = "relu_{}".format(i)
			elif isinstance(layer, nn.MaxPool2d):
				name = "pool_{}".format(i)
			elif isinstance(layer, nn.BatchNorm2d):
				name = "bn_{}".format(i)
			else:
				raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

			model.add_module(name, layer)

			if name in self.content_layers:
				# add content loss:
				content_feature = model(content_img).detach()
				content_loss = ContentLoss(content_feature)
				model.add_module("content_loss_{}".format(i), content_loss)
				self.content_losses.append(content_loss)

			if name in self.style_layers:
				# add style loss:
				style_feature = model(style_img).detach()
				style_loss = StyleLoss(style_feature)
				model.add_module("style_loss_{}".format(i), style_loss)
				self.style_losses.append(style_loss)

		self.model = model

	def run(self, num_steps):
		"""
		Run the style transfer
		"""
		def closure():
			# correct the values
			input_img.data.clamp_(0, 1)

			optimizer.zero_grad()

			self.model(input_img)

			content_score = 0
			style_score = 0

			for cl in self.content_losses:
				content_score += cl.loss
			#weighing loss on the content
			content_score *= content_weight

			for sl in self.style_losses:
				style_score += sl.loss
			#weighing loss on the style
			style_score *= style_weight

			loss = content_score + style_score
			loss.backward()

			self.step += 1
			if self.step % 50 == 0:
				style_score_val = style_score.item()
				# print(log_pattern.format(self.step, content_score.item(), style_score_val))

				k = (self.prev_score + smooth)/(style_score_val + smooth)
				if abs(1-k) > eps:
					self.prev_score = style_score_val
				else:
					self.step = 1e10

			return loss

		content_weight, style_weight = 1, 1e5
		# log_pattern = "Step: {}\tContent Loss: {:.4f} Style Loss: {:.4f}"
		eps = 2e-1
		smooth = 1e-4

		self.prev_score = float("inf")

		input_img = self.content_img.clone().detach()
		# this line to show that input is a parameter that requires a gradient
		optimizer = optim.LBFGS([input_img.requires_grad_()])

		self.step = 1
		while self.step < num_steps:
			optimizer.step(closure)

		# a last correction...
		input_img.data.clamp_(0, 1)

		return input_img.detach()