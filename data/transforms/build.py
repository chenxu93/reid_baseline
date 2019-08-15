import torchvision.transforms as T
from .transforms import RandomErasing
import random


def build_transforms(cfg, phase='train'):
	normalize = T.Normalize(mean=cfg.DATASET.IMG_MEAN, std=cfg.DATASET.IMG_STD)

	if phase == "train":
		compose_list = [
				T.Resize(cfg.DATASET.SIZE_TRAIN),
				T.RandomHorizontalFlip(p=cfg.DATASET.FILP_P),
				T.Pad(cfg.DATASET.IMG_PADDING_PIXEL),
				T.RandomCrop(cfg.DATASET.SIZE_TRAIN),
			]

		if random.uniform(0,1) < cfg.DATASET.ROTATE_P:
			compose_list += [T.RandomRotation(degrees=cfg.DATASET.RANDOM_ROTATE_ANGLE)]

		compose_list += [
				T.ToTensor(),
				normalize,
				RandomErasing(probability=cfg.DATASET.RANDOM_ERASE_P, mean=cfg.DATASET.IMG_MEAN)
			]
		transform = T.Compose(compose_list)
	elif phase ==  "test":
		compose_list = [
				T.Resize(cfg.DATASET.SIZE_TEST),
				T.ToTensor(),
				normalize
		]
		transform = T.Compose(compose_list)
	else:
		raise ValueError("expect 'train' or 'test' phase, but got '{}'".fotmat(phase))

	return transform