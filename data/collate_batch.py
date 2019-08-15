import torch

def train_collate_fn(batch):
	imgs, pids, camids, imgs_path = zip(*batch)
	pids = torch.tensor(pids, dtype=torch.int64)

	return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
	imgs, pids, camids, imgs_path = zip(*batch)

	return torch.stack(imgs, dim=0), pids, camids