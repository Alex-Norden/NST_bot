import sys
import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


DEFINED_STYLES = (
	"candy",
	"composition_vii",
	"escher_sphere",
	"feathers",
	"frida_kahlo",
	"la_muse",
	"mosaic",
	"mosaic_ducks_massimo",
	"pencil",
	"picasso_selfport",
	"rain_princess",
	"robert_delaunay",
	"seated_nude",
	"shipwreck",
	"starry_night",
	"stars",
	"strip",
	"the_scream",
	"udnie",
	"wave",
	"woman_with_hat_matisse")

STYLE_CMDS = {f"/{name}": i for i, name in enumerate(DEFINED_STYLES)}


class PATH:
	ROOT_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))

	MODEL1_PATH = os.path.join(ROOT_DIR, "resources/Gatys.model")
	MODEL2_PATH = os.path.join(ROOT_DIR, "resources/21styles.model")

	DEFINED_STYLES_DIR = os.path.join(ROOT_DIR, "resources/21styles")
	USERDATA_DIR = os.path.join(ROOT_DIR, "userdata")

	CONTENT_PATH = os.path.join(USERDATA_DIR, "content.jpg")
	STYLE_PATH = os.path.join(USERDATA_DIR, "style.jpg")
	RESULT_PATH = os.path.join(USERDATA_DIR, "result.jpg")


# ---------------------- IMAGE PROCESSING ----------------------------
def resize_images(img1, img2, target_size):
	img1_source_size = img1.size

	scale = target_size / max(img1_source_size)
	img1 = img1.resize((round(img1_source_size[0] * scale), round(img1_source_size[1] * scale)), Image.ANTIALIAS)

	scale = target_size / min(img2.size)
	img2 = img2.resize((round(img2.size[0] * scale), round(img2.size[1] * scale)), Image.ANTIALIAS)

	return img1, img2, img1_source_size

def load_images_norm(target_size):
	def to_tensor(img):
		img = loader(img).unsqueeze(0)
		return img.to(torch.float)

	img1, img2, img1_source_size = resize_images(Image.open(PATH.CONTENT_PATH),
												Image.open(PATH.STYLE_PATH),
												target_size)

	loader = transforms.Compose([transforms.CenterCrop((img1.size[1], img1.size[0])),
								transforms.ToTensor()])

	return to_tensor(img1), to_tensor(img2), img1_source_size

def load_images_rgb(style_num, target_size):
	def to_tensor(img):
		img = loader(img)
		img = np.array(img).transpose(2, 0, 1)
		return torch.from_numpy(img).float().unsqueeze(0)

	style_path = os.path.join(PATH.DEFINED_STYLES_DIR, f"{DEFINED_STYLES[style_num]}.jpg")

	img1, img2, img1_source_size = resize_images(Image.open(PATH.CONTENT_PATH).convert("RGB"),
													Image.open(style_path).convert("RGB"),
													target_size)

	loader = transforms.Compose([transforms.CenterCrop((img1.size[1], img1.size[0]))])

	return to_tensor(img1), to_tensor(img2), img1_source_size


def save_image(img, sorce_size):
	img = img.resize(sorce_size, Image.ANTIALIAS)
	img.save(PATH.RESULT_PATH)

unloader = transforms.ToPILImage()
def save_image_norm(tensor, sorce_size):
	tensor = tensor.clone().squeeze(0)
	img = unloader(tensor)
	save_image(img, sorce_size)

def save_image_rgb(tensor, sorce_size):
	tensor = tensor.clone().squeeze(0)

	(b, g, r) = torch.chunk(tensor, 3)
	tensor = torch.cat((r, g, b))

	img = tensor.clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype("uint8")
	img = Image.fromarray(img)

	save_image(img, sorce_size)


def preprocess_batch(batch):
	batch = batch.transpose(0, 1)
	(r, g, b) = torch.chunk(batch, 3)
	batch = torch.cat((b, g, r))
	batch = batch.transpose(0, 1)
	return batch