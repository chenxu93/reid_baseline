import torch
from torch import nn

class CenterLoss(nn.Module):
	"""
	center loss
	reference:
		Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

	Args:
	cfg: config file
	num_class(int): the calsses number of training samples
	
	"""
	def __init__(self, cfg, num_class):
		super(CenterLoss, self).__init__()
		self.num_classes = num_class
		self.feature_dims = cfg.MODEL.FEATURE_DIMS
		self.use_gpu = cfg.SOLVER.USE_GPU
		self.gpu_id = cfg.SOLVER.GPU_ID

		if self.use_gpu:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dims).cuda(self.gpu_id))
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dims))

	def forward(self, features, targets):
		"""
		Args:
			features(torch FloatTensor): model output features in conv5 layers, with shape [batch_size, feature_dims]
			targets(torch LongTensot): ground truth, with shape [num_classes]
		"""

		assert features.size(0) == targets.size(0), "features size: {} and labels size: {} unmatch".format(self.feature_dims.size(0), targets.size(0))
		batch_size = features.size(0)
		# ||A - B||**2 = ||A||**2 + ||B||**2 - 2*A.B
		dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
					torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		dist_mat.addmm_(1, -2, features, self.centers.t()) # shape[batch_size, batch_size]

		classes = torch.arange(self.num_classes).long()
		if self.use_gpu: 
			classes = classes.cuda(self.gpu_id)
		labels = targets.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = []
		for i in range(batch_size):
			value = dist_mat[i][mask[i]]
			value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
			dist.append(value)
		dist = torch.cat(dist)
		loss = dist.mean()

		return loss
