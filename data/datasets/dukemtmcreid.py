from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import glob
import re

from .base import BaseDataset

class DukeMTMCreID(BaseDataset):
	"""docstring for DukeMTMCreID"""
	def __init__(self, cfg):
		super(DukeMTMCreID, self).__init__()
		self.dataset_dir = cfg.DATASET.ROOT
		if not osp.exists(self.dataset_dir):
			raise ValueError("dataset path not exists: check given path {} ".format(self.dataset_dir))

		self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
		self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
		self.query_dir = osp.join(self.dataset_dir,"query")

		self.check_path()

		self.train = self.process_dir(self.train_dir, relabel=True) #[(img_path, pid, camid), ..., (img_path, pid, camid)]
		self.gallery = self.process_dir(self.gallery_dir, relabel=False)
		self.query = self.process_dir(self.query_dir, relabel=False)

		self.num_train_imgs, self.num_train_pids, self.num_train_cams = self.get_imageitem_info(self.train)
		self.num_gallery_imgs, self.num_gallery_pids, self.num_gallery_cams = self.get_imageitem_info(self.gallery)
		self.num_query_imgs, self.num_query_pids, self.num_query_cams = self.get_imageitem_info(self.query)


	def process_dir(self, dir_path, relabel=False):
		img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
		pattern = re.compile(r'([-\d]+)_c(\d)')

		pid_container = set()
		for img_path in img_paths:
			pid, _ = map(int, pattern.search(img_path).groups())
			pid_container.add(pid)
		pid2label = {pid:label for label, pid in enumerate(pid_container)}

		data = []
		for img_path in img_paths:
			pid, camid = map(int, pattern.search(img_path).groups())
			assert 1 <= camid <= 8
			camid -= 1 # index from 0
			if relabel:
				pid = pid2label[pid]
			data.append((img_path, pid, camid))

		return data