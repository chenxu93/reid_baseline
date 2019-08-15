from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import os.path as osp
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .avgmeter import AverageMeter
from .enginetools import save_checkpoint
from metrics.distance import compute_matrix_distance
from metrics.rank import evaluate_rank
from reid.utils.rerank import re_ranking
from reid.utils.tools import visualize_ranked_results



class Engine(object):
	"""docstring for Engine"""
	def __init__(self, model, optimizer=None, schedule=None):
		
		self.model = model
		self.schedule = schedule

		# check model
		if not isinstance(self.model, nn.Module):
			raise TypeError("model must be an instance of nn.Moudle, got '{}'".format(type(model)))

	def run(self, cfg, dataset, train_loader, query_loader, gallery_loader, optimizer, scheduler, loss_fn):


		if cfg.TRAIN.TEST_ONLY:
			self.test(cfg, 0, dataset, query_loader, gallery_loader)

			return

		# start training ...
		start_time = time.time()
		print("\nstart training ......")
		max_rank = 0.
		max_mAP = 0.
		rank1 = 0.
		
		is_best = False

		for epoch in range(cfg.SOLVER.START_EPOCH, cfg.SOLVER.MAX_EPOCH):
			self.train(cfg, epoch, train_loader)

			# eval during training ...
			if cfg.TRAIN.IF_EVAL and (epoch + 1) >= cfg.TRAIN.START_EVAL and (epoch + 1)%cfg.TRAIN.EVAL_FREQ == 0:
				cmc, mAP = self.test(cfg, epoch, dataset, query_loader, gallery_loader)
				rank1 = cmc[0]

				if rank1 > max_rank and mAP > max_mAP:
					max_rank = rank1
					max_mAP = mAP
					is_best = True

			# save trained model
			if (epoch + 1)%cfg.SOLVER.SAVE_FREQ == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCH:
				self._save_checkpoint(cfg, epoch, is_best, remove_module_from_keys=False)
				if is_best:
					# save best model
					# TODO

					is_best = False

		end_time = round(time.time() - start_time)
		end_time = str(datetime.timedelta(seconds=end_time))

		print("training done, consume: {}".format(end_time))

		# perform evaluate
		print("performing evaluate on the dataset .....")
		cmc, mAP = self.evaluate(cfg, dataset, query_loader, gallery_loader, epoch=cfg.SOLVER.MAX_EPOCH,
			dist_metric=cfg.TEST.DIST_METRIC, normalize_feature=cfg.TEST.FEATURE_NORMALIZE, rerank==cfg.TEST.RE_RANKING, 
			visrank=cfg.TEST.VISRANK, visrank_topk=cfg.TEST.VISRANK_TOPK, use_metric_cuhk03=cfg.TEST.METRIC_CUHK03, ranks=cfg.TEST.RANKS)

	def train():
		"""
		perform traing on dataset for one epoch,
		this function will be called in run() function, like below code block:
		###
			for epoch in range(start_epoch, max_epoch):
				self.train(agrs)

		
		###		
		
		This train() function should be implemented in each subclass.

		"""

	def test(self, cfg, epoch, dataset, query_loader, gallery_loader, ):
		"""
		test model performance, with query and gallery
		Args:
			epoch(int): current epoch
			dataset(classes): dataset classes
			query_loader(nn.Dataloader): query dataloader 
			gallery_loader(nn.Dataloader): gallery dataloader

		return rakn-1 cmc
		"""
		dist_metric = cfg.TEST.DIST_METRIC
		normalize_feature = cfg.TEST.FEATURE_NORMALIZE
		visrank = cfg.TEST.VISRANK
		visrank_topk = cfg.TEST.VISRANK_TOPK
		use_metric_cuhk03 = cfg.TEST.METRIC_CUHK03
		ranks = cfg.TEST.RANKS
		rerank = cfg.TEST.RE_RANKING

		cmc, mAP = self.evaluate(
				cfg,
				dataset,
				query_loader,
				gallery_loader,
				epoch,
				dist_metric,
				normalize_feature,
				rerank,
				visrank,
				visrank_topk,
				use_metric_cuhk03,
				ranks
			)

		return cmc, mAP

	@torch.no_grad()
	def evaluate(self, cfg, dataset, query_loader=None, gallery_loader=None, epoch=0,
			dist_metric='euclidean', normalize_feature=False, rerank=False, visrank=False, 
			visrank_topk=20, use_metric_cuhk03=False, ranks=[1, 5, 10, 20]):
		batch_time = AverageMeter()

		self.model.eval()

		print("\nextracting features from query dataset ...")
		query_features, query_pids, query_camids = [], [], [] ## query features, query person IDs and query camera IDs

		for batch_idx, data in enumerate(query_loader):
			imgs, pids, camids = self.parse_data_for_eval(data)
			if cfg.SOLVER.USE_GPU:
				imgs = imgs.cuda(cfg.SOLVER.GPU_ID)
			end = time.time()
			features = self.model(imgs)
			batch_time.update(time.time() - end)
			features = features.data.cpu()

			query_features.append(features)
			query_pids.extend(pids)
			query_camids.extend(camids)

		query_features = torch.cat(query_features, 0)
		query_pids = np.asarray(query_pids)
		query_camids = np.asarray(query_camids)
		print("\nextract query features done, obtained {}-by-{} matrix"
			.format(query_features.size(0), query_features.size(1)))
		
		print("\nextracting features from gallery dataset ...")
		gallery_features, gallery_pids, gallery_camids = [], [], [] ## gallery features, gallery person IDs and gallery camera IDs
		end = time.time()
		for batch_idx, data in enumerate(gallery_loader):
			imgs, pids, camids = self.parse_data_for_eval(data)
			if cfg.SOLVER.USE_GPU and torch.cuda.is_available():
				imgs = imgs.cuda(cfg.SOLVER.GPU_ID)
			end = time.time()
			features = self.model(imgs)
			batch_time.update(time.time() - end)
			features = features.data.cpu()
			gallery_features.append(features)
			gallery_pids.extend(pids)
			gallery_camids.extend(camids)

		gallery_features = torch.cat(gallery_features, 0)
		gallery_pids = np.asarray(gallery_pids)
		gallery_camids = np.asarray(gallery_camids)
		print("\nextract gallery features done, obtained {}-by-{} matrix"
			.format(gallery_features.size(0), gallery_features.size(1)))

		print("cost time: {:.4f}s,  avg: {:.4f} s/batch".format(batch_time.val, batch_time.avg))

		if normalize_feature:
			print("perform L2 norm for query and gallery features ...")
			query_features = F.normalize(query_features, p=2, dim=1)
			gallery_features = F.normalize(gallery_features, p=2, dim=1)

		print("computing distance with '{}' metric ...".format(dist_metric))
		dist_mat = 	compute_matrix_distance(query_features, gallery_features, dist_metric)
		dist_mat = dist_mat.numpy()

		if rerank:
			print("\nperform re-ranking ...")
			dist_mat_query = compute_matrix_distance(query_features, gallery_features, dist_metric)
			dist_mat_gallery = compute_matrix_distance(gallery_features, gallery_features, dist_metric)
			dist_mat = re_ranking(dist_mat, dist_mat_query, dist_mat_gallery)

		print("computing CMC and mAP ...")
		cmc, mAP = evaluate_rank(dist_mat, query_pids, gallery_pids, query_camids, gallery_camids, use_metric_cuhk03=use_metric_cuhk03)

		print("\n*************** results ***************")
		print("\t mAP: {:.2%}".format(mAP))
		print("\t CMC curve: ")
		for r in ranks:
			print("\t Rank-{}: {:.2%}".format(r, cmc[r-1]))
		print("\n****************************************")

		if visrank:
			# TODO
			visualize_ranked_results(dist_mat, (dataset.query, dataset.gallery), save_dir=osp.join(save_dir,'visrank-'+str(epoch+1)), topk=visrank_topk)

		return cmc, mAP
			


	def _save_checkpoint(self, cfg, epoch, is_best=False, remove_module_from_keys=False):
		save_checkpoint(cfg,
			{
			'state_dict': self.model.state_dict(),
			'epoch': epoch + 1,
			'optimizer': self.optimizer.state_dict()

			},
			epoch,
			is_best=is_best,
			remove_module_from_keys=remove_module_from_keys)
	
	def parse_data_for_train(self, data):
		imgs = data[0]
		pids = data[1]

		return imgs, pids

	def parse_data_for_eval(self, data):
		imgs = data[0]
		pids = data[1]
		camids = data[2]

		return imgs, pids, camids

	def extract_features(self, x):
		self.model.eval()
		return self.model(x)


