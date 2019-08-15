
import torch
import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyWithLabelSmooth
from .center_loss import CenterLoss


def build_criterion(cfg, num_classes):

	center_criterion = None
	loss_type = cfg.TRAIN.LOSS_TYPE
	
	use_label_smooth = cfg.TRAIN.LABEL_SMOOTH


	if 'softmax' != loss_type[:7]:
		raise ValueError("must use the softmax to train the classifier, input loss_type: '{}'".format(loss_type))

	if 'triplet' in loss_type:
		triplet = TripletLoss(cfg)
	if 'center' in loss_type:
		center_criterion = CenterLoss(cfg, num_classes)
	if 'range' in loss_type:
		# TODO
		pass
	if 'cluster' in loss_type:
		# TODO
		pass

	def loss_with_label_smooth(scores, features, targets):
		# is use label smooth
		xent = CrossEntropyWithLabelSmooth(cfg, num_classes=num_classes)
		

		if loss_type == 'softmax_triplet':
			return  xent(scores, targets) + cfg.LOSS.TRIPLET_LOSS_WEIGHT * triplet(features, targets)
		elif loss_type == 'softmax_center':
			return  xent(scores, targets) + cfg.LOSS.CENTER_LOSS_WEIGHT * center_criterion(features, targets)
		elif loss_type == 'softmax_triplet_center' or loss_type == 'softmax_center_triplet':
			return xent(scores, targets) + cfg.LOSS.TRIPLET_LOSS_WEIGHT * triplet(features, targets) + \
											cfg.LOSS.CENTER_LOSS_WEIGHT * center_criterion(features, targets)
		else:
			raise ValueError("loss type error, got: '{}'".format(loss_type))

	def loss_without_label_smooth(scores, features, targets):

		if loss_type == 'softmax_triplet':
			return  F.cross_entropy(scoresm, targets) + cfg.LOSS.TRIPLET_LOSS_WEIGHT * triplet(features, targets)
		elif loss_type == 'softmax_center':
			return  F.cross_entropy(scoresm, targets) + cfg.LOSS.CENTER_LOSS_WEIGHT * center_criterion(features, targets)
		elif loss_type == 'softmax_triplet_center' or loss_type == 'softmax_center_triplet':
			return F.cross_entropy(scoresm, targets) + cfg.LOSS.TRIPLET_LOSS_WEIGHT * triplet(features, targets) + \
														cfg.LOSS.CENTER_LOSS_WEIGHT * center_criterion(features, targets)
		else:
			raise ValueError("loss type error, got: '{}'".format(loss_type))


	def loss_func(scores, features, targets):
		"""
		return a loss function 
		Args：
			socres(torch Tensor): model predict scores, with shape [abtch_size, num_classes]
			features(torch FloatTensor): global features
			targets(torch LongTensor): ground truth, with shape [abtch_size,]
		"""

		if cfg.SOLVER.USE_GPU and torch.cuda.is_available():
			targets = targets.cuda(cfg.SOLVER.GPU_ID)

		if not cfg.DATALOADER.USE_IDENTITY_SAMPLER or loss_type == 'softmax':
			return F.cross_entropy(scores, targets)

		if use_label_smooth:
			# use label smooth 
			return loss_with_label_smooth(scores, features, targets)

		else:
			# do not use label smooth
			return loss_without_label_smooth(scores, features, targets)

	return loss_func, center_criterion


