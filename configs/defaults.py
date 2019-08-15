from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from future.utils import iteritems
import copy
import io
import logging
import numpy as np
import os
import os.path as osp
import six
import yaml
# import reid.utils.collections
from reid.utils.collections import AttrDict


logger = logging.getLogger(__name__)
"""
--------------------------------------------
	tradin and test paraments setting
--------------------------------------------
"""

__C = AttrDict()
cfg = __C

# --------------------------------------------
# basic optins
# --------------------------------------------

#random seed
__C.RNG_SEED = 1



# --------------------------------------------
# model options
# --------------------------------------------
__C.MODEL = AttrDict()

# 
__C.MODEL.TYPE = 'trick_baseline'

__C.MODEL.BACKBONE_CHOICES = ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
__C.MODEL.BACKBONE = 'resnet50'

__C.MODEL.LAST_STRIDE = 2

# if use bnneck layer, choice: ('no','bnneck')
__C.MODEL.BN_NECK_CHOICES = ('no', 'bnneck')
# if add bnneck layer 
__C.MODEL.BN_NECK = 'bnneck'
# feature dimmensions, 512 for resnet18 and resnet 34, 2048 for resnet50, resnet101, resnet152.
__C.MODEL.FEATURE_DIMS = 2048
__C.MODEL.USE_BN_NECK_FEATURE = True



# --------------------------------------------
# training options
# --------------------------------------------

__C.TRAIN = AttrDict()
# pretrained model path for training
__C.TRAIN.WEIGHTS = ''
# train classes, (deprecated)
__C.TRAIN.NUM_CLASSES = 1000

__C.TRAIN.FC_DIMS = None
# maximum epoch for training
__C.TRAIN.DROPOUT_P = None
# number of epochs to train 'open_layers' (new layers),  
# while keeping base layers frozen. Default is 0. 'fixbase_epoch' is counted in 'max_epoch'
__C.TRAIN.FIX_BASE_EPOCH = 0
#layers (attribute names) open for training.
__C.TRAIN.OPEN_LAYERS = None #(str or list, optional)
#from which epoch to start evaluation. Default is 0.
__C.TRAIN.START_EVAL = 0
__C.TRAIN.IF_EVAL = False # if False, means evaluation is only performed at the end of training
# evaluation frequency, only useful when IF_EVAL is 'True' 
__C.TRAIN.EVAL_FREQ = 10

# if True, only runs evaluation on test datasets.
__C.TRAIN.TEST_ONLY = False 


# performs L2 normalization on feature vectors before computing feature distance. Default is True.
__C.TRAIN.NORM_FEATURE = True

__C.TRAIN.LABEL_SMOOTH = True
# whether remove "modeule." in model state_dict's keys() when save modeel during training
__C.TRAIN.REMOVE_MODULE_FROM_KEYS = False





# --------------------------------------------
# test options
# --------------------------------------------
__C.TEST = AttrDict()
# test batch size
__C.TEST.BATCH_SIZE = 128
# perform re-rank during evaluate or test
__C.TEST.RE_RANKING = False # 'no' # false in deep-reid
# test trained model 
__C.TEST.WEIGHTS = ''
# whether use BN neck feature during evaluate or test
__C.TEST.NECK_FEATURE = 'after'
# perform L2 normalize for output feature
__C.TEST.FEATURE_NORMALIZE = True
# whether visualize rank during evauate or test
__C.TEST.VISRANK = False
# visualize topk rank, only useful when VISRANK is 'True'
__C.TEST.VISRANK_TOPK = 20
# 
__C.TEST.SAVE_DIR = './visrank'
# whether use cuhk03 evaluate for evaluating
__C.TEST.METRIC_CUHK03 = False
# ranks topk
__C.TEST.RANKS = [1, 5, 10, 20]
# distance metric used to compute distance matrix between query and gallery. options: ['euclidean', 'cosine']Default is "euclidean".
__C.TEST.DIST_METRIC = 'euclidean'
# pretrained model path for testing
__C.TEST.WEIGHTS = ''


# --------------------------------------------
# 
# --------------------------------------------

__C.DATASET = AttrDict()

__C.DATASET.ROOT = ''

# dataset name: include [Market1501, Market1501_500k, ...]
__C.DATASET.NAMES = ('market1501',)

__C.DATASET.MARKET1501_500K = False

__C.DATASET.SIZE_TRAIN = [384, 128]

__C.DATASET.SIZE_TEST = [384, 128]

__C.DATASET.FILP_P = 0.5

__C.DATASET.RANDOM_ERASE_P = 0.5

__C.DATASET.ROTATE_P = 0.5

__C.DATASET.RANDOM_ROTATE_ANGLE = 5

__C.DATASET.IMG_MEAN = [0.485, 0.456, 0.406]

__C.DATASET.IMG_STD = [0.229, 0.224, 0.225]

__C.DATASET.IMG_PADDING_PIXEL = 10


# --------------------------------------------

# --------------------------------------------
__C.DATALOADER = AttrDict()

__C.DATALOADER.NUM_WORKERS = 0
# sampler type, if True, RandomIdentitySampler are select, otherwise sampler is common random sampler
__C.DATALOADER.USE_IDENTITY_SAMPLER = True

__C.DATALOADER.NUM_INSTALCE = 16


# --------------------------------------------

# --------------------------------------------

__C.SOLVER = AttrDict()
# optimizer type, name follow totch.optim
__C.SOLVER.OPTIMIZER = 'SGD'
# weight decay
__C.SOLVER.WEIGHT_DECAY = 5e-04
#
__C.SOLVER.MOMENTUM = 0.9
#
__C.SOLVER.BIAS_LR_FACTOR = 2
#
__C.SOLVER.WEIGHT_DECAY_BIAS = 0.


__C.SOLVER.BASE_LR = 0.0001
# learning rate decay gamma
__C.SOLVER.GAMMA = 0.1

__C.SOLVER.BATCH_SIZE = 64

# starting epoch. Default is 0.
__C.SOLVER.START_EPOCH = 0
# total training epoch 
__C.SOLVER.MAX_EPOCH = 0
# print_frequency. Default is 10.
__C.SOLVER.PRINT_FREQ = 10

__C.SOLVER.SAVE_FREQ = 1
#
__C.SOLVER.WARMUP_LR = True
#
__C.SOLVER.LR_SCHEDULE = 'step'
# warmup learning schedule, 
__C.SOLVER.WARMUP_METHOD = 'constant' # 'constant' or linear
# decay step of learning rate
__C.SOLVER.STEPS = [30, 55]
#
__C.SOLVER.WARMUP_FACTOR = 1.0 / 3
#
__C.SOLVER.WARMUP_ITERS = 10
#
__C.SOLVER.LAST_EPOCH = -1
# top k accuracy 
__C.SOLVER.TOPK = [1, ]
# whether use gpu on the model
__C.SOLVER.USE_GPU = True
# gpu id
__C.SOLVER.GPU_ID = 1
# center loss learning rate
__C.SOLVER.CENTER_LR = 0.5


# --------------------------------------------
# for evaluate metric
# --------------------------------------------



# --------------------------------------------
# loss function
# --------------------------------------------
__C.LOSS = AttrDict()
# loss type, incluing [softmax, triple, softmax_triplet, crossentropy_labelsmooth, ]
__C.TRAIN.LOSS_TYPE = 'softmax'

__C.LOSS.TRIPLET_MARGIN = 0.3

__C.LOSS.LABEL_SMOOTH_EPSILON = 0.1
# balance the feature(triplet loss) loss and classsification loss
__C.LOSS.TRIPLET_LOSS_WEIGHT = 1.0
# whether perform hard example mining on triplet loss
__C.LOSS.HARD_EXAMPLE_MINING = False
# balance the feature(center loss) loss and classsification loss
__C.LOSS.CENTER_LOSS_WEIGHT = 0.0005

# --------------------------------------------
# resume from a trained model
# --------------------------------------------
__C.RESUME = AttrDict()

__C.RESUME.IF_RESUME = False
#
__C.RESUME.RESUME_FROM = '.'

# --------------------------------------------
# output dir
# --------------------------------------------
__C.OUTPUT = AttrDict()

__C.OUTPUT.SAVE_MODEL_PATH = '.'

_RENAMED_MODULES = {
    'utils.collections': 'reid.utils.collections',
}


def load_cfg(cfg_to_load):
	"""Wrapper around yaml.load used for maintaining backward compatibility"""
	file_types = [file, io.IOBase] if six.PY2 else [io.IOBase]  # noqa false positive
	expected_types = tuple(file_types + list(six.string_types))
	assert isinstance(cfg_to_load, expected_types), \
		'Expected one of {}, got {}'.format(expected_types, type(cfg_to_load))
	if isinstance(cfg_to_load, tuple(file_types)):
		cfg_to_load = ''.join(cfg_to_load.readlines())
	for old_module, new_module in iteritems(_RENAMED_MODULES):
		# yaml object encoding: !!python/object/new:<module>.<object>
		old_module, new_module = 'new:' + old_module, 'new:' + new_module
		cfg_to_load = cfg_to_load.replace(old_module, new_module)
	return yaml.load(cfg_to_load)


def merge_cfg_from_file(cfg_filename):
	"""Load a yaml config file and merge it into the global config."""
	with open(cfg_filename, 'r') as f:
		yaml_cfg = AttrDict(load_cfg(f))
	_merge_a_into_b(yaml_cfg, __C)

def merge_cfg_from_cfg(cfg_other):
	"""Merge `cfg_other` into the global config."""
	_merge_a_into_b(cfg_other, __C)

def merge_cfg_from_list(cfg_list):
	"""Merge config keys, values in a list (e.g., from command line) into the
	global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
	"""
	assert len(cfg_list) % 2 == 0
	for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):

		key_list = full_key.split('.')
		d = __C
		for subkey in key_list[:-1]:
			assert subkey in d, 'Non-existent key: {}'.format(full_key)
			d = d[subkey]
		subkey = key_list[-1]
		assert subkey in d, 'Non-existent key: {}'.format(full_key)
		value = _decode_cfg_value(v)
		value = _check_and_coerce_cfg_value_type(
			value, d[subkey], subkey, full_key
		)
		d[subkey] = value
		
def _merge_a_into_b(a, b, stack=None):
	"""Merge config dictionary a into config dictionary b, clobbering the
	options in b whenever they are also specified in a.
	"""
	assert isinstance(a, AttrDict), \
		'`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
	assert isinstance(b, AttrDict), \
		'`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

	for k, v_ in a.items():
		full_key = '.'.join(stack) + '.' + k if stack is not None else k
		# a must specify keys that are in b
		if k not in b:
			raise KeyError('Non-existent config key: {}'.format(full_key))

		v = copy.deepcopy(v_)
		v = _decode_cfg_value(v)
		v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

		# Recursively merge dicts
		if isinstance(v, AttrDict):
			try:
				stack_push = [k] if stack is None else stack + [k]
				_merge_a_into_b(v, b[k], stack=stack_push)
			except BaseException:
				raise
		else:
			b[k] = v

def _decode_cfg_value(v):
	"""Decodes a raw config value (e.g., from a yaml config files or command
	line argument) into a Python object.
	"""
	# Configs parsed from raw yaml will contain dictionary keys that need to be
	# converted to AttrDict objects
	if isinstance(v, dict):
		return AttrDict(v)
	# All remaining processing is only applied to strings
	if not isinstance(v, six.string_types):
		return v
	# Try to interpret `v` as a:
	#   string, number, tuple, list, dict, boolean, or None
	try:
		v = literal_eval(v)
	# The following two excepts allow v to pass through when it represents a
	# string.
	#
	# Longer explanation:
	# The type of v is always a string (before calling literal_eval), but
	# sometimes it *represents* a string and other times a data structure, like
	# a list. In the case that v represents a string, what we got back from the
	# yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
	# ok with '"foo"', but will raise a ValueError if given 'foo'. In other
	# cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
	# will raise a SyntaxError.
	except ValueError:
		pass
	except SyntaxError:
		pass
	return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
	"""Checks that `value_a`, which is intended to replace `value_b` is of the
	right type. The type is correct if it matches exactly or is one of a few
	cases in which the type can be easily coerced.
	"""
	# The types must match (with some exceptions)
	type_b = type(value_b)
	type_a = type(value_a)
	if type_a is type_b:
		return value_a

	# Exceptions: numpy arrays, strings, tuple<->list
	if isinstance(value_b, np.ndarray):
		value_a = np.array(value_a, dtype=value_b.dtype)
	elif isinstance(value_b, six.string_types):
		value_a = str(value_a)
	elif isinstance(value_a, tuple) and isinstance(value_b, list):
		value_a = list(value_a)
	elif isinstance(value_a, list) and isinstance(value_b, tuple):
		value_a = tuple(value_a)
	else:
		raise ValueError(
			'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
			'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
		)
	return value_a

def assert_and_infer_cfg(make_immutable=True):
	"""Call this function in your script after you have finished setting all cfg
	values that are necessary (e.g., merging a config from a file, merging
	command line config options, etc.). By default, this function will also
	mark the global cfg as immutable to prevent changing the global cfg settings
	during script execution (which can lead to hard to debug errors or code
	that's harder to understand than is necessary).
	"""
	if make_immutable:
		cfg.immutable(True)