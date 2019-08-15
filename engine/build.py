from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .engine import Engine 
from .train_imgs import ImageEngine

all_phases = ['train', 'test', 'run']

class BuildEngine(object):
	"""
	build engine for training or test model

	Args:
		datamanager:(tuple of list): contain dataset, train_loader, query_loader and gallery_loader,
				is not provided, use 'None' as placeholder 
		model(nn.Module): model for training or testing
		criterion(): loss type
		optimizer(nn.optim): optimizer
		schuedule(): learning scheudle
		phase(str): train, test or run are useful. mean the model used for training, testing or training&testing respectively
	"""
	def __init__(self, cfg, datamanager, model, criterion=None, optimizer=None, scheudler=None, 
					center_criterion=None, optimizer_center=None, phase='run'):
		super(BuildEngine, self).__init__()
		self.cfg = cfg
		self.datamanager = datamanager
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheudler = scheudler
		self.phase = phase
		self.use_gpu = self.cfg.SOLVER.USE_GPU
		

	def run(self):
		"""
		perform train and test for the model
		"""
		assert self.phase == 'run', "engin phase except 'run', but got: {}".format(self.phase)
		engine = ImageEngine(self.cfg, self.model, self.criterion, self.optimizer, self.scheudler)
		engine_run = engine.run(self.cfg, self.datamanager[0], self.datamanager[1],
							self.datamanager[2], self.datamanager[3], self.optimizer, self.scheudler, self.criterion)
		return engine_run


	def train(self):
		"""
		perform train for the model
		"""
		assert self.phase == 'train', "engin phase except 'train', but got: {}".format(self.phase)
		# engine = ImageEngine(self.cfg, self,model, self.criterion, self.optimizer, self.scheudle)
		# TODO


	def test(self):
		"""
		perform test for the model
		"""
		# TODO
		assert self.phase == 'test', "engin phase except 'run', but got: {}".format(self.phase)
		engin = Engine(self.model)


		return engin
		
