from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs.defaults import cfg
"""
reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch 
import torch.nn as nn
def conv3x3(in_channels, out_channels, stride=1):
	"""
	convolution 3*3 module, with padding = 1 and no bias	
	"""
	conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
			padding=1, bias=False)
	return conv_3x3

def conv1x1(in_channels, out_channels):
	""" 1*1 convolution"""
	conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
	return conv_1x1


class BasicBlock(nn.Module):
	""" bottleneck building block for ResNet18 abd ResNet34"""
	expansion = 1
	def __init__(self, in_channels, mid_channels, stride=1, downsample=None,):
		super(BasicBlock, self).__init__()

		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(in_channels, mid_channels, stride)
		self.bn1 = norm_layer(in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(mid_channels, mid_channels)
		self.bn2 = norm_layer(mid_channels)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity =x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)


		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	""" bottleneck building block for ResNet50, ResNet101 and ResNet 152"""
	expansion = 4
	def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
		super(Bottleneck, self).__init__()

		self.conv1 = conv1x1(in_channels, mid_channels)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.conv2 = conv3x3(mid_channels, mid_channels, stride)
		self.bn2 = nn.BatchNorm2d(mid_channels)
		self.conv3 = conv1x1(mid_channels, mid_channels * self.expansion)
		self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	"""
	docstring for ResNet
	Args:

	"""

	def __init__(self, block, layers=[3, 4, 6, 3], last_stride=2):
		super(ResNet, self).__init__()

		self.in_channels = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
								bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1],stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

		
	def _make_layer(self, block, out_channels, blocks, stride=1):
		"""
		construct an layer, the downsample is used to reduce channel dims with conv1x1 between 
		tow "bottleneck" building block rather than the feature map size
		"""
		downsample = None

		if stride != 1 or self.in_channels != out_channels * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, 
							stride=stride, bias=False),
				nn.BatchNorm2d(out_channels * block.expansion),
				)

		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels * block.expansion

		for _ in range(1,blocks):
			layers.append(block(self.in_channels, out_channels))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x





	def random_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) :
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def load_parameters(self, model_path):
		params_dict = torch.load(model_path)
		for i in params_dict:
			if 'fc' in i:
				continue
			self.state_dict()[i].copy_(params_dict[i])

		




