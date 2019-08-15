import torch
import torch.nn as nn

def normalize(x, p=2, axis=-1, e=10e-12):
	"""
	perform Lp normalize on tensor x alone dim axis
	"""
	x = 1. * x / (torch.norm(x, p, axis, keepdim=True).expand_as(x) + e)
	return x

def euclidean_dist(x, y):
	"""
	calculate euclidean distance between tensor x and y
	dist = sqrt(||x - y||**2 = ||x||**2 + ||y||** - 2*xy)
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
	"""
	for each anchro find the hardest positive and negative sample.
	Args:
		dist_max(torch Variable): pair wise distance between samples, shape [N, N]
		labels(torch LongTensor): shape [N]
		return_inds(bool): whether to return indices. save time if 'False'
	return:
		dist_ap(torch Variable): distance(anchor, positive); shape [N]
		dist_an(torch Variable): distance(anchor, negative); shape [N]
		p_inds(torch LongTensor): indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1. shape [N]
		n_inds(torch LongTensor): indices of selected hard negative samples; 0 <= p_inds[i] <= N - 1. shape [N]

	NOTE: Only consider the case in which all labels have same num of samples,
			thus we can cope with all anchors in parallel.

	"""
	assert len(dist_mat.size()) == 2, "the dimensions of distance matrix must queal 2"
	assert dist_mat.size(0) == dist_mat.size(1), "dim 0 and dim 1 must be queal in distance matrix"

	N = dist_mat.size(0)
	# shape[N,N]
	is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
	is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

	# 'dist_ap' means distance(anchor, positive)
	# both 'dist_ap' and 'relative_p_inds' with shape [N, 1]
	dist_ap, relative_p_inds = torch.max(
		dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

	# 'dist_an' means distance(anchor, negative)
	# both 'dist_an' and 'relative_n_inds' with shape [N, 1]
	dist_an, relative_n_inds = torch.min(
		dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
	# shape [N]
	dist_ap = dist_ap.squeeze(1)
	dist_an = dist_an.squeeze(1)

	if return_inds:
		# shape [N, N]
		ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
		# shape [N, 1]
		p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
		n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
		# shape [N]
		p_inds = p_inds.squeeze(1)
		n_inds = n_inds.squeeze(1)

		return dist_ap, dist_an, p_inds, n_inds

	return dist_ap, dist_an


class TripletLoss(nn.Module):
	"""
	Reference:
		Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

	Modified from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`


	
	"""
	def __init__(self, cfg):
		super(TripletLoss, self).__init__()
		self.margin = cfg.LOSS.TRIPLET_MARGIN
		self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
		self.norm_feature = cfg.TRAIN.NORM_FEATURE
		self.sampler = cfg.DATALOADER.USE_IDENTITY_SAMPLER # if true, 'RandomIdentitySampler' is selected

	def forward(self, global_feat, labels):
		if self.norm_feature:
			global_feat = normalize(global_feat, axis=-1)

			dist_mat = euclidean_dist(global_feat, global_feat)

		if self.sampler:
			# perform hard example mining for triplet loss
			# this hard example mining require sampler is RandomIdentitySampler
			dist_ap, dist_an = hard_example_mining(dist_mat, labels)
			y = dist_an.new().resize_as_(dist_an).fill_(1)
		else:
			# this is also a hard example mining for triplet loss, but this not require RandomIdentitySampler
			n = global_feat.size(0)
			# for each anchor find the hardest positive anchor and negative
			mask = labels.expand(n, n).eq(labels.expand(n, n).t())
			dist_ap, dist_an = [] ,[]
			for i in range(n):
				dist_ap.append(dist_mat[i][mask[i]].max().unsqueeze(0))
				dist_an.append(dist_mat[i][mask[i] == 0].min().unsqueeze(0))
			dist_ap = torch.cat(dist_ap)
			dist_an = torch.cat(dist_an)
			y = torch.ones_like(dist_an)

		if self.margin > 0.:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)

		return loss


class CrossEntropyWithLabelSmooth(nn.Module	):
	"""
	Cross entropy loss with label smoothing regularizer.

	Args:
		num_classes(int): number classes of training samples 
		epsilon (float): parameter of LabelSmooth (discard)
	"""
	def __init__(self, cfg, num_classes, use_gpu=True):
		super(CrossEntropyWithLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = cfg.LOSS.LABEL_SMOOTH_EPSILON
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)
		self.gpu_id = cfg.SOLVER.GPU_ID


	def forward(self, predicts, targets):
		"""
		Args:
			preidcts(torch Tensor): the predict matrix with shape (batch_size, num_classes)
		"""
		log_probs = self.logsoftmax(predicts)

		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) # onehot encoding

		if self.use_gpu: 
			targets = targets.cuda(self.gpu_id)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()

		return loss


		


