
from reid.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn
from configs.defaults import cfg
from reid.utils.tools import check_available_file
from reid.utils.tools import weights_init_kaiming
from reid.utils.tools import weights_init_classifier
class TricksNet(nn.Module):
	"""docstring for TricksNet
	implement paper: http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
	"""
	in_channels = 2048
	def __init__(self, cfg, num_classes, neck, phase='train'):
		super(TricksNet, self).__init__()
		if cfg.MODEL.BACKBONE == 'resnet18':
			self.in_channels = 512
			self.backbone = ResNet(
					block=BasicBlock,
					layers=[2, 2, 2, 2],
					last_stride=cfg.MODEL.LAST_STRIDE
				)
		elif cfg.MODEL.BACKBONE == 'resnet34':
			self.in_channels = 512
			self.backbone = ResNet(
					block=BasicBlock,
					layers=[3, 4, 6, 3],
					last_stride=cfg.MODEL.LAST_STRIDE
				)
		elif cfg.MODEL.BACKBONE == 'resnet50':
			self.backbone = ResNet(
					block=Bottleneck,
					layers=[3, 4, 6, 3],
					last_stride=cfg.MODEL.LAST_STRIDE
				)
		elif cfg.MODEL.BACKBONE == 'resnet101':
			self.backbone = ResNet(
					block=Bottleneck,
					layers=[3, 4, 23, 3],
					last_stride=cfg.MODEL.LAST_STRIDE
				)
		elif cfg.MODEL.BACKBONE == 'resnet152':
			self.backbone = ResNet(
					block=Bottleneck,
					layers=[3, 8, 38, 3],
					last_stride=cfg.MODEL.LAST_STRIDE
				)
		else:
			raise ValueError('(Bag of Tricks Algorithm): expect a model backbone type in "{}", but got "{}"'.format(
							cfg.MODEL.BACKBONE_CHOICES, cfg.MODEL.BACKBONE))
		if cfg.TRAIN.WEIGHTS != '' and check_available_file(cfg.TRAIN.WEIGHTS) and phase == 'train':
			self.backbone.load_parameters(cfg.TRAIN.WEIGHTS)
			print("load pretrained model patameters from: {}".format(cfg.TRAIN.WEIGHTS))
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

		self.num_classes = num_classes
		self.bnneck = cfg.MODEL.USE_BN_NECK_FEATURE
		self.phase = phase

		if self.bnneck:
			# use bnneck layer
			print("\nuse bnneck for training and testing ...")
			self.bottleneck = nn.BatchNorm1d(self.in_channels)
			self.bottleneck.bias.requires_grad_(False) 
			self.classifier = nn.Linear(self.in_channels, self.num_classes, bias=False)
			if phase == 'train':
				self.bottleneck.apply(weights_init_kaiming)
			
			
		else:
			print("\ndo not use bnneck for training and testing ...")
			self.classifier = nn.Linear(self.in_channels, self.num_classes)

		if phase == 'train':
			self.classifier.apply(weights_init_classifier)


	def forward(self, x):
		global_feature = self.global_avg_pool(self.backbone(x)) # (N, in_channels, 1, 1)
		global_feature = global_feature.view(global_feature.size(0), -1) # (N, 2048)

		if self.bnneck:
			feature = self.bottleneck(global_feature)
		else:
			feature = global_feature

		if self.phase == 'train':
			cls_score = self.classifier(feature)
			return cls_score, global_feature
		elif self.phase == 'test':
			if self.bnneck:
				return feature
			else:
				return global_feature

	def load_parameters(self,model_path):
		params_dict = torch.load(model_path)
		for i in params_dict:
			self.state_dict()[i].copy_(params_dict[i])


