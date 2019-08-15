'''
@ author: xuchen
@ email: upchen19@foxmail.com
'''
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17


CUHK03 = None


img_datasets = {
	"market1501": Market1501,
	'cuhk03': CUHK03,
	'dukemtmcreid': DukeMTMCreID,
	'msmt17': MSMT17,
}

def init_dataset(cfg, name, *args, **kwargs):
	if name not in img_datasets.keys():
		raise KeyError('dataset "{}" not found, optional choices: "{}".'.format(name, img_datasets.keys()))
	return img_datasets[name](cfg, *args, **kwargs)