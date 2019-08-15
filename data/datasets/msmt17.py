from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import glob
import re

from .base import BaseDataset


# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred

TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
			'MSMT17_V1': {TRAIN_DIR_KEY: 'train', TEST_DIR_KEY: 'test'},
			'MSMT17_V2': {TRAIN_DIR_KEY: 'mask_train_v2', TEST_DIR_KEY: 'mask_test_v2'}
		}

class MSMT17(BaseDataset):
	"""docstring for MSMT17"""
	def __init__(self, cfg, combine_all=False):
		super(MSMT17, self).__init__()
		self.dataset_dir = cfg.DATASET.ROOT
		if not osp.exists(self.dataset_dir):
			raise ValueError("dataset path not exists: check given path {} ".format(self.dataset_dir))

		has_main_dir = False
		dataset_name = None
		for name in VERSION_DICT.keys():
			if osp.exists(osp.join(self.dataset_dir, name)):
				train_dir = VERSION_DICT[name][TRAIN_DIR_KEY]
				test_dir = VERSION_DICT[name][TEST_DIR_KEY]
				has_main_dir = True
				dataset_name = name
				break

		assert has_main_dir and dataset_name is not None, "dataset folder not found"

		self.train_dir = osp.join(self.dataset_dir, dataset_name, train_dir)
		self.test_dir = osp.join(self.dataset_dir, dataset_name, test_dir)
		self.list_train_path = osp.join(self.dataset_dir, dataset_name, 'list_train.txt')
		self.list_val_path = osp.join(self.dataset_dir, dataset_name, 'list_val.txt')
		self.list_query_path = osp.join(self.dataset_dir, dataset_name, 'list_query.txt')
		self.list_gallery_path = osp.join(self.dataset_dir, dataset_name, 'list_gallery.txt')

		if not osp.exists(self.train_dir):
			raise ValueError("train folder: '{}' do not exists".format(self.train_dir))
		if not osp.exists(self.train_dir):
			raise ValueError("test folder: '{}' do not exists".format(self.test_dir))

		self.train = self.process_dir(self.train_dir, self.list_train_path)
		self.val = self.process_dir(self.train_dir, self.list_val_path)
		self.query = self.process_dir(self.test_dir, self.list_query_path)
		self.gallery = self.process_dir(self.test_dir, self.list_gallery_path)

		if combine_all:
			self.train += self.val

		self.num_train_imgs, self.num_train_pids, self.num_train_cams = self.get_imageitem_info(self.train)
		self.num_gallery_imgs, self.num_gallery_pids, self.num_gallery_cams = self.get_imageitem_info(self.gallery)
		self.num_query_imgs, self.num_query_pids, self.num_query_cams = self.get_imageitem_info(self.query)

	def process_dir(self, dir_path, list_path):
		with open(list_path, 'r') as f:
			lines = f.readlines()

		data = []

		for img_idx, img_info in enumerate(lines):
			img_path, pid = img_info.split(' ')
			pid = int(pid) # do not need relabel
			camid = int(img_path.split('_')[2]) - 1 # index bengin with 0
			img_path = osp.join(dir_path, img_path)
			data.append((img_path, pid, camid))

		return data
