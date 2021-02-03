import threading
import traceback
import time
import logging

import torch
from torch.autograd import Variable

import config
from utils import PATH, load_images_norm, load_images_rgb, save_image_norm, save_image_rgb, preprocess_batch
from net_first import GatysNet, GatysTransfer
from net_second import MSGNet


LOG = logging.getLogger(__name__)


class TaskThread(threading.Thread):

	def __init__(self):
		super(TaskThread,self).__init__()

		self.setDaemon(True)
		self.running = None

		self.has_task = None
		self.is_first = None
		self.style_num = None

		self.is_success = None
		self.last_task_id = 0

		self._init_models()

	def _init_models(self):
		def init_first():
			model = GatysNet()
			model.load_state_dict(torch.load(PATH.MODEL1_PATH))
			self.transfer1 = GatysTransfer(model.features.eval())

		def init_second():
			self.transfer2 = MSGNet(ngf=128)

			model_dict = torch.load(PATH.MODEL2_PATH)
			model_dict_clone = model_dict.copy()
			for key, value in model_dict_clone.items():
				if key.endswith(("running_mean", "running_var")):
					del model_dict[key]

			self.transfer2.load_state_dict(model_dict, False)

		init_first()
		init_second()

	def transfer(self):
		if self.is_first:
			content_img, style_img, content_source_size = load_images_norm(config.TARGET_SIZE1)
			# stylize
			self.transfer1.build_model(content_img, style_img)
			stylized_img = self.transfer1.run(config.NUM_STEPS)

			save_image_norm(stylized_img, content_source_size)
		else:
			content_img, style_img, content_source_size = load_images_rgb(self.style_num, config.TARGET_SIZE2)
			# stylize
			style_var = Variable(preprocess_batch(style_img))
			content_var = Variable(preprocess_batch(content_img))

			self.transfer2.setTarget(style_var)
			stylized_img = self.transfer2(content_var).detach()

			save_image_rgb(stylized_img, content_source_size)

		return True

	def run(self):
		self.running = True

		while self.running:
			if self.has_task:
				try:
					self.is_success = self.transfer()
				except Exception:
					LOG.critical(f"[TaskThread] exception: {traceback.format_exc()}")
					self.is_success = False
				self.has_task = False
			else:
				time.sleep(1.0)

		self.running = False