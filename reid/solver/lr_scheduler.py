from bisect import bisect_right
import torch

available_schedule = ["step", "multi_step"]

def build_lr_scheduler(optimizer, lr_scheduler, step_size, gamma=0.1):

	if lr_scheduler not in available_schedule:
		raise TypeError("learning rate scheduler type error, except '{}', but got '{}'".format(available_schedule, lr_scheduler))

	if lr_scheduler == 'step':
		if isinstance(step_size, (list, tuple)):
			step_size = step_size[-1]

		if not isinstance(step_size, int):
			raise TypeError("for step learning rate scheduler, step size must be a type of integer, but got {}".format(type(step_size)))

		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)	
	elif lr_scheduler == 'multi_step':
		if not isinstance(step_size, (list, tuple)):
			raise TypeError("for multi_step learning scheduler, multi_step must be a type of list or tuple, but got {}".format(type(step_size)))

		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma)

	return scheduler	
			


def adjust_learning_rate(cfg, optimizer, index):
	"""
	Sets the learning rate to the initial LR decayed by 10 at every specified step
	# Adapted from PyTorch Imagenet example: https://github.com/pytorch/examples/blob/master/imagenet/main.py

	"""
	base_lr = cfg.SOLVER.BASE_LR
	gamma = cfg.SOLVER.BASE_LR.GAMMA 

	lr = base_lr * (gamma **  index)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
	"""
	
	"""
	def __init__(
		self, 
		optimizer,
		milestones,
		gamma=0.1,
		warmup_factor=1.0 / 3,
		warmup_iters=500,
		warmup_method="linear",
		last_epoch=-1,
		):
		
		if not list(milestones) == sorted(milestones):
			raise ValueError(
					"Milestones should be a list of" " increasing integers. Got {}",
					milestones,
				)
		if warmup_method not in ("constant", "linear"):
			raise ValueError(
					 "Only 'constant' or 'linear' warmup_method accepted"
					 "got {}".format(warmup_method)
				)
			
		self.milestones = milestones
		self.gamma = gamma
		self.warmup_factor = warmup_factor
		self.warmup_iters = warmup_iters
		self.warmup_method = warmup_method
		super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		warmup_factor = 1
		if self.last_epoch < self.warmup_iters:
			if self.warmup_method == "constant":
				warmup_factor = self.warmup_factor
			elif self.warmup_method == "linear":
				alpha = self.last_epoch / self.warmup_iters
				warmup_factor = self.warmup_factor * (1 - alpha) + alpha

		return [
			base_lr
			* warmup_factor
			* self.gamma ** bisect_right(self.milestones, self.last_epoch)
			for base_lr in self.base_lrs
		]
		