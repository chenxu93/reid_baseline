from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import shutil
import warnings
import os
import os.path as osp
from functools import partial
import pickle

import torch
import torch.nn as nn




def save_checkpoint(cfg,state, epoch=0,is_best=False, remove_module_from_keys=False):
	if not osp.exists(cfg.OUTPUT.SAVE_MODEL_PATH):
		os.makedirs(cfg.OUTPUT.SAVE_MODEL_PATH)

	if remove_module_from_keys:
		# remove 'module.' in dict's keys 
		state_dict = state['state_dict']
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			if k.startwith('module.'):
				k = k[7:]
			new_state_dict[k] = k
		state['state_dict'] = new_state_dict

	# save
	fpath = osp.join(cfg.OUTPUT.SAVE_MODEL_PATH, cfg.MODEL.TYPE + '_epoch{}.pth'.format(epoch+1))
	torch.save(state['state_dict'], fpath)
	print("========== save checkpoint file to: '{}' ==========".format(fpath))
	if is_best:
		shutil.copy(fpath, osp.join(osp.dirname(fpath), "model_best.pth"))


def load_checkpoint(fpath):
	if fpath is None:
		raise ValueError('File path do nor found at: "{}"'.format(fpath))
	map_location = None if torch.cuda.is_available() else 'cpu'

	try:
		checkpoint = torch.load(fpath, map_location=map_location)
	except UnicodeDecodeError:
		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
	except Exception:
		print("can not load checkpoint from '{}'".format(fpath))
		raise
	return checkpoint


def resume_from_checkpoint(fpath, model, optimizer=None):
	"""
	resume tringing from a pretrained checkpoint
	this function will load :
		① model weights
		② state_dict of optimizer if 'optimizer' is not None

	Args:
		fpath(str): checkpoint path
		model(nn.Module): train model
		optimizer(Optimizer, optional): Optimizer

	return int: start epoch
	"""

	pass

def open_all_layers(model):
	model.train()
	for p in model.parameters():
		p.requires_grad = True


def open_specified_layers(cfg, model):
	open_layers = cfg.TRAIN.OPEN_LAYERS
	if isinstance(model, nn.DataParallel):
		model = model.module

	if isinstance(cfg.TRAIN.OPEN_LAYERS, str):
		open_layers = [open_layers]

	for layer in open_layers:
		assert hasattr(model, layer), "'{}' is not a layer of the model".format(layer)

	for name, module in model.named_children():
		if name in open_layers:
			module.train()
			for p in module.parameters():
				p.requires_grad = True
		else:
			module.eval()
			for p in module.parameters():
				p.requires_grad = False


def load_pretrained_weights(model, weight_path):
	"""
	load pretrained parameters to model
	Features:
		-- Incompatible layers (unmatched in name or size) will be ignored.
		-- Can automatically deal with keys containing "module.".

	Args:
		model(nn.Module): model network
		weight_path(srt): pretrained model file path
	"""

	checkpoint = load_checkpoint(weight_path)
	if 'state_dict' in checkpoint:
		state_dict = checkpoint['state_dict']
	else:
		state_dict = checkpoint

	model_dict = model.state_dict()
	new_state_dict = OrderedDict()
	matched_layers, discarded_layers = [], []

	for k, v in state_dict.items():
		if k.startwith('module.'):
			k = k[7:]

		if k in model_dict and model_dict[k].size() == v.size():
			new_state_dict[k] = v
			matched_layers.append(k)
		else:
			discarded_layers.append(k)

	model_dict.update(new_state_dict)
	model.load_state_dict(model_dict)

	if len(matched_layers) == 0:
		warnings.warn(
			"the pretrained model '{}' can not be loaded, "
			"you should check the modle keys manually first, ('ignored and continue')".format(weight_path)
			)
	else:
		print("load pretrained model weight done! model path: '{}'".format(weight_path))
		if len(discarded_layers) > 0:
			print("******************************************")
			print("the follow layer in pretrained model '{}' has been discarded \
			 due to keys or layer size unmatch with training network".format(discarded_layers))
			print("******************************************")


