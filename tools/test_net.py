import os
import os.path as osp
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


from data.build import make_data_loader

from engine.build import BuildEngine


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

	logger.info('Testing with config:')
	logger.info(pprint.pformat(cfg))
	np.random.seed(cfg.RNG_SEED)

	model_path = cfg.TEST.WEIGHTS
	if not osp.exists(osp.dirname(model_path)) and osp.isfile(model_path):
		raise TypeError("the pretrained model is invalid, check '{}' befor test the model".format(model_path))

	# build dataset 
	dataset, train_loader, query_loader, gallery_loader, num_query, num_classes = make_data_loader(cfg)
	datamanager = (dataset, train_loader, query_loader, gallery_loader)
	# buile model ...
	model = TricksNet(cfg, num_classes=num_classes, neck=cfg.MODEL.BN_NECK, phase="test")

	if cfg.SOLVER.USE_GPU:
		model = model.cuda(cfg.SOLVER.GPU_ID)
	# load parameters from pretrained model
	print("loading pretrained model from {}".format(model_path))
	model.load_parameters(model_path)

	engine = BuildEngine(cfg, datamanager, model, phase='test').test()
	engine.evaluate(cfg, dataset, query_loader, gallery_loader, dist_metric=cfg.TEST.DIST_METRIC, normalize_feature=cfg.TEST.FEATURE_NORMALIZE,
					rerank=cfg.TEST.RE_RANKING, use_metric_cuhk03=cfg.TEST.METRIC_CUHK03, ranks=cfg.TEST.RANKS)
	print("done!")

	

if __name__ == '__main__':
	main()