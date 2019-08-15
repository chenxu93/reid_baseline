from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


def accuracy(predicts, targets, topk=(1,)):
	"""
	compute model predict top-k accuracy

	Args:
		predicts(torch Tensor): model output scores tensor, matrix shoape [batch_size, num_classes]
		target(torch LongTensor): ground truth labels, with shape [batch_size]

	return:
		list: top-k accuracy

	"""
	maxk = max(topk)
	batch_size = targets.size(0)

	if isinstance(predicts, (list, tuple)):
		predicts = predicts[0]

	probs, pred_labels = predicts.topk(maxk, 1, largest=True, sorted=True)

	pred_labels = pred_labels.t() # shape [1, batch_size]
	correct = pred_labels.eq(targets.view(1, -1).expand_as(pred_labels)) # shape [1, batch_size]

	rst = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		acc = correct_k.mul_(1. / batch_size)
		rst.append(acc)

	return rst