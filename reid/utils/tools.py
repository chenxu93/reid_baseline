
import torch
import torch.nn as nn

import os
import os.path as osp
import shutil
import numpy as np 

def check_available_file(file_path):
	isfile = osp.isfile(file_path)
	if not isfile:
		print("FIle Not Exists Error: '{}' not Found. ".format(file_path))\
		
	return isfile


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)


def visualize_ranked_results(dist_mat, dataset, save_dir='.', topk=5):
	"""
	visualizes ranked results 
	Args:
		dist_mat(numpu.ndarray): distance matrix with shape [num_query, num_gallery].
		dataset(tuple): a 2-tuple containing (query, gallery), each of which contains
			tuples of (img_path(s), pid, camid).
		save_dir(str): directory to save output images.
		topk(int): denoting top-k images in the rank list to be visualized.

	"""
	num_query, num_gallery = dist_mat.shape

	print("visualizing top-{} ranks".format(topk))
	print("\n********** query images: {} ,"
		"gallery images: {} **********\n".format(num_gallery, num_gallery))

	query, gallery = dataset
	assert num_query == len(query)
	assert num_gallery == len(gallery)

	indices = np.argsort(dist_mat, axis=1) # sort by distance, and return index
	if not osp.exists(save_dir):
		os.makedirs(save_dir)

	def cp_img_to(src, dst, rank, prefix):
		"""
		copy test images to specified path

		Args:
			src(str): test images path
			dst(str): destination path of visualize path
			prefix(str): prefix
		"""

		if isinstance(src, (tuple, list)):
			dst = osp.join(dst, prefix + "_top" + str(rank))
			if not osp.exists(dst):
				os.makedirs(dst)
				for img_path in src:
					shutil.copy(img_path, dst)
		else:
			dst = osp.join(dst, prefix + "_top" + str(rank)+ "__" + osp.basename(src))
			shutil.copy(src, dst)

	for query_idx in range(num_query):
		query_img_path, query_pid, query_camid = query[query_idx]
		if isinstance(query_img_path (list, tuple)):
			query_dir = osp.join(save_dir, osp.basename(query_img_path[0]))
		else:
			query_dir = osp.join(save_dir, osp.basename(query_img_path))

		if not osp.exists(query_dir):
			os.makedirs(query_dir)
		cp_img_to(query_img_path,query_dir, rank=0, prefix='query')
		
		rank_idx = 1
		for gallery_idx in indices[query_idx, :]:
			gallery_img_path, gallery_pid, gallery_camid = gallery[gallery_idx]
			invalid = (query_pid == gallery_pid) & (query_camid == gallery_camid)
			if not invalid:
				cp_img_to(gallery_img_path, query_dir, rank=rank_idx, prefix='gallery')
				rank_idx += 1
				if rank_idx > topk:
					break

	print("copy image done!")		


