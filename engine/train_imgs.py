from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch
import numpy as np

from .engine import Engine

from .avgmeter import AverageMeter
from reid.solver.lr_scheduler import adjust_learning_rate

import metrics

class ImageEngine(Engine):
	"""docstring for ImageEngine"""
	def __init__(self,cfg, model, criterion, optimizer, scheduler=None, center_criterion=None, optimizer_center=None):
		super(ImageEngine, self).__init__(model, optimizer, scheduler)
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.max_epoch = cfg.SOLVER.MAX_EPOCH
		self.scheduler =scheduler
		self.center_criterion = center_criterion
		self.optimizer_center = optimizer_center


	def train(self, cfg, epoch, train_dateloader):
		losses = AverageMeter()
		acc = AverageMeter()
		batch_time = AverageMeter()
		data_time = AverageMeter()

		self.model.train()
		"""
		do some frozen layer option
		"""
		end = time.time()
		for step, data in enumerate(train_dateloader):
			data_time.update(time.time() - end)

			imgs, pids = self.parse_data_for_train(data)
			
			if cfg.SOLVER.USE_GPU and torch.cuda.is_available():
				# load data into GPU device
				imgs = imgs.cuda(cfg.SOLVER.GPU_ID)
				pids = pids.cuda(cfg.SOLVER.GPU_ID)

			targets = pids
			self.optimizer.zero_grad()
			if self.optimizer_center is not None:
				self.optimizer_center.zero_grad()

			# forward 
			scores, features = self.model(imgs)
			loss = self.criterion(scores, features, targets)
			
			# back propagation
			loss.backward()
			self.optimizer.step()
			if self.optimizer_center is not None:
				for parm in self.center_criterion.parameters():
					param.grad.data *= (1. / cfg.SOLVER.CENTER_LR)
				self.optimizer_center.step()

			# update AverageMeter
			losses.update(loss.item(), targets.size(0))
			# acc.update((scores.max(1)[1] == targets).float().mean())

			acc.update(metrics.accuracy(scores, targets)[0].item()) # use top 1 accuracy
			batch_time.update(time.time() - end)

			# print traing info
			if(step + 1) % cfg.SOLVER.PRINT_FREQ == 0:
				num_batches = len(train_dateloader)
				eta_seconds = batch_time.avg * (num_batches - (step + 1) + (self.max_epoch - (epoch + 1)) * num_batches)
				eta = str(datetime.timedelta(seconds=int(eta_seconds)))

				print("epoch: [{}/{}] || "
					"iter: [{}/{}] || "
					"batch time: {:.3f} || "
					"loss: {:.4f} || "
					"accuracy: {:.4f}% || "
					"lr: {:.8f} || "
					"eta: {}"
					.format(
						(epoch + 1), self.max_epoch,
						step + 1, num_batches,
						batch_time.avg,
						losses.avg,
						acc.avg * 100,
						self.optimizer.param_groups[0]['lr'], 
						eta
					))
			end = time.time()
		self.scheduler.step()

			


