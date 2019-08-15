
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset

from .datasets.dataset_loader import ImageDataset
from .samplers.sampler import RandomIdentitySampler
from .transforms.build import build_transforms


def make_data_loader(cfg):
	train_transform = build_transforms(cfg, phase='train')
	val_transform = build_transforms(cfg, phase='test')

	num_workers = cfg.DATALOADER.NUM_WORKERS

	if len(cfg.DATASET.NAMES) == 1:
		dataset = init_dataset(cfg, cfg.DATASET.NAMES[0])
	elif len(cfg.DATASET.NAMES) > 1:
		# TODO: add multi dataset to train
		raise NotImplementedError("train on multiple datasets is not implemented!")
	else:
		raise ValueError("not give train dataset name, check 'DATASET.NAMES' paramenter")
		

	num_classes = dataset.num_train_pids

	train_set = ImageDataset(dataset.train, train_transform)

	# if use identity sampler
	if cfg.DATALOADER.USE_IDENTITY_SAMPLER:
		train_loader = DataLoader(
							train_set, batch_size=cfg.SOLVER.BATCH_SIZE, 
							sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.BATCH_SIZE, cfg.DATALOADER.NUM_INSTALCE),
							num_workers=num_workers, collate_fn=train_collate_fn,drop_last=True
						)
	else:
		train_loader = DataLoader(
							train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=num_workers,
							collate_fn=train_collate_fn,drop_last=True
						)

	query_set = ImageDataset(dataset.query, val_transform)
	gallery_set = ImageDataset(dataset.gallery, val_transform)

	query_loader = DataLoader(
			query_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=num_workers,
			collate_fn=val_collate_fn
		)

	gallery_loader = DataLoader(
			gallery_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=num_workers, 
			collate_fn=val_collate_fn
		)

	print("Dataset Stastistic, name: '{}'".format(cfg.DATASET.NAMES))
	print("---------------------------------------------------------------------------------")
	print("	subset	|	# ids 	|	# images 	|	# cameras ")
	print("---------------------------------------------------------------------------------")
	print("	train 	|	{:6d}	|	{:8d}	| {:9d}".format(dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams))
	print("	query 	|	{:6d}	|	{:8d}	| {:9d}".format(dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams))
	print("	gallery |	{:6d}	|	{:8d}	| {:9d}".format(dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams))
	print("---------------------------------------------------------------------------------")


	return dataset, train_loader, query_loader, gallery_loader, len(dataset.query), num_classes