import os.path as osp
from PIL import Image 
from torch.utils.data import Dataset

class ImageDataset(Dataset):
	"""ImageDataset"""
	def __init__(self, dataset,transform=None):
		super(ImageDataset, self).__init__()
		self.dataset = dataset # process_dir function: (img_path, pid, camid)
		self.transform = transform

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		img_path, pid, camid = self.dataset[idx]
		img = Image.open(img_path)
		if self.transform is not None:
			img = self.transform(img)
		
		return img, pid, camid, img_path

	

