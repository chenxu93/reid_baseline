import os.path as osp


class BaseDataset(object):
	"""docstring for BaseDataset"""
	def get_imageitem_info(self, items):
		pids, cams = [], []
		for _, pid, camid in items:
			pids += [pid]
			cams += [camid]
		
		pids = set(pids)
		cams = set(cams)

		num_pids = len(pids)
		num_cams = len(cams)
		num_imgs = len(items)

		return num_imgs, num_pids, num_cams 


	def check_path(self):
		if not osp.exists(self.train_dir):
			raise RuntimeError('train path: {} do not exists.'.format(self.train_dir))
		if not osp.exists(self.gallery_dir):
			raise RuntimeError('gallery path: {} do not exists.'.format(self.gallery_dir))
		if not osp.exists(self.query_dir):
			raise RuntimeError('query path: {} do not exists.'.format(self.query_dir))