from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np 
import pprint
import argparse

import torch
import torch.nn as nn

from configs.defaults import cfg
from configs.defaults import merge_cfg_from_file, merge_cfg_from_list
from configs.defaults import assert_and_infer_cfg
from reid.utils.logging import setup_logging

from reid.models.tricks_net import TricksNet

from reid.solver import adjust_learning_rate, WarmupMultiStepLR, make_optimizer, build_lr_scheduler
# --------------------------------
from data.build import make_data_loader

from engine.build import BuildEngine

from losses.build import build_criterion

def parse_args():
	parser = argparse.ArgumentParser(description="train a reid model")
	parser.add_argument(
		"--cfg",
		dest='cfg_file',
		default=None,
		help="see config xx.yaml file",
		type=str
		)
	parser.add_argument(
		'opts',
		help="see reid config default.py for all options",
		default=None,
		nargs=argparse.REMAINDER
		)

	return parser.parse_args()


def main():
	logger = setup_logging(__name__)
	logging.getLogger('reid.loader').setLevel(logging.INFO)

	args = parse_args()
	logger.info('Called with args:')
	logger.info(args)
	# print("config fiel path: {}".format(args.cfg_file))
	if args.cfg_file is not None:
		merge_cfg_from_file(args.cfg_file)
	if args.opts is not None:
		merge_cfg_from_list(args.opts)

	assert_and_infer_cfg()

	logger.info('Training with config:')
	logger.info(pprint.pformat(cfg))
	
	np.random.seed(cfg.RNG_SEED)

	dataset, train_loader, query_loader, gallery_loader, num_query, num_classes = make_data_loader(cfg)
	datamanager = (dataset, train_loader, query_loader, gallery_loader)
	criterion, center_criterion = build_criterion(cfg, num_classes)
	if center_criterion is not None:
		logger.info("\ntraining with center loss, ceneter loss lr: {}".format(cfg.SOLVER.CENTER_LR))
	else:
		logger.info("\ntraining without center loss ... ")

	if cfg.TRAIN.LABEL_SMOOTH:
		logger.info("\nuse label smooth for training, epsilon parameter: {}, on {} classes".format(cfg.LOSS.LABEL_SMOOTH_EPSILON, num_classes))
	else:
		logger.info("\ndo not use label smooth for training")

	model = TricksNet(cfg, num_classes=num_classes, neck=cfg.MODEL.BN_NECK, phase="train")

	if cfg.SOLVER.USE_GPU:
		model = model.cuda(cfg.SOLVER.GPU_ID)
	print("load model done!")

	if cfg.RESUME.IF_RESUME:
		print("wait to impement")
	else:
		# center optimizer use SGD optimizer with constant learing rate ... 
		optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
		if cfg.SOLVER.WARMUP_LR:
			scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, 
											cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
		else:
			scheduler = build_lr_scheduler(optimizer, cfg.SOLVER.LR_SCHEDULE, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)

	
	# (self, cfg, datamanager, model, criterion, optimizer, scheudle, phase='run'):
	engine = BuildEngine(cfg, datamanager, model, criterion, optimizer, scheduler, center_criterion, optimizer_center, phase='run')
	engine.run()

if __name__ == '__main__':
	main()
	


