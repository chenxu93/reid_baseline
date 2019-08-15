from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
from torch.nn import functional as F

def compute_matrix_distance(x, y, metric='euclidean'):
	"""
	compute distance between x and y with metric

	Args:
		x(torch Tensor): 2-D features matrix, with shape [batch_size, feature_dims]
		y(torch Tensor): 2-D features matrix, with shape [batch_size, feature_dims]
		metric(str): distance metric (euclidean or cosine)
	"""

	assert isinstance(x, torch.Tensor)
	assert isinstance(y, torch.Tensor)
	assert (x.dim() == 2 and y.dim() ==2), "Expected 2-D tensor, but got '{}'-D and '{}'-D".format(x.dim(), y.dim())
	assert x.size(1) == y.size(1), "features dimensin must euqal, but got '{}' and '{}' respectively".format(x.size(1), y.size(1))

	if metric == 'euclidean':
		dist_mat = euclidean_squared_distance(x, y)
	elif metric == 'cosine':
		dist_mat = cosine_distance(x, y)
	else:
		raise ValueError("except 'euclidean' or 'cosine' metric, but got '{}'".format(metric))

	return dist_mat


def euclidean_squared_distance(x, y):
	"""
	compute euclidean squared distance between x and y

	Args:
		x(torch Tensor): 2-D features matrix, with shape [m, feature_dims]
		y(torch Tensor): 2-D features matrix, with shape [n, feature_dims]
	return:
		torch Tensor: distance matrix

	"""

	m, n = x.size(0), y.size(0)
	# distance(x,y): ||x - y||**2 = ||x||**2 + ||y||**2 - 2*xy
	dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m,n) + \
				torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n,m).t()

	dist_mat.addmm_(1, -2, x, y.t())		

	return dist_mat

def cosine_distance(x, y):
	"""
	compute cosine distance between x and y

	Args:
		x(torch Tensor): 2-D features matrix, with shape [m, feature_dims]
		y(torch Tensor): 2-D features matrix, with shape [n, feature_dims]
	return:
		torch Tensor: distance matrix
	"""

	x_norm = F.normalize(x, p=2, dim=1)
	y_norm = F.normalize(y, p=2, dim=1)

	dist_mat = 1 - torch.mm(x_norm, y_norm.t())

	return dist_mat