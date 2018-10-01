""" ============== Super Resolution Dense Network ============ """
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
	def __init__(self, num_input_features, growth_rate, bot_neck, drop_rate):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_features = num_input_features)),
		self.add_module('leaky_relu1', nn.LeakyReLU(inplace = True)),
		self.add_module('conv1', nn.Conv2d(in_channels = num_input_features, out_channels = bot_neck * growth_rate, kernel_size = 1, stride = 1, bias = False)),
		self.add_module('norm2', nn.BatchNorm2d(num_features = bot_neck * growth_rate)),
		self.add_module('leaky_relu2', nn.LeakyReLU(inplace = True)),
		self.add_module('conv2', nn.Conv2d(in_channels = bot_neck * growth_rate, out_channels = growth_rate, kernel_size = 3, stride = 1, padding = 1, bias = False)),
		self.drop_rate = drop_rate

	def forward(self, x):
		new_features = super(_DenseLayer, self).forward(x)
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p = self.drop_rate, training = self.training)
		return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
	def __init__(self, num_layers, num_input_features, growth_rate, bot_neck, drop_rate):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bot_neck, drop_rate)
			self.add_module('denselayer%d' % (i+1), layer)

"""
======= No Transition Layer =======
class _TransitionLayer(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_TransitionLayer, self).__init__()
		self.add_module('g_norm', nn.GroupNorm(num_groups = num_input_features // 2, num_channels = num_input_features))
		self.add_module('leaky_relu', nn.LeakyReLU(inplace  = True))
		self.add_module('conv', nn.Conv2d(in_channels = num_input_features, out_channels = num_output_features, kernel_size = 1, stride = 1, bias = False))
		self.add_module('avg_pool', nn.AvgPool2d(kernel_size = 2, stride = 2))
"""

class _UpsampleBlock(nn.Module):
	""" Designed following input from paper and 
    	link https://distill.pub/2016/deconv-checkerboard/
    """
	def __init__(self, in_channels, middle_channels, out_channels):
		super(_UpsampleBlock, self).__init__()
		self.in_channels = in_channels
		self.block = nn.Sequential(
			nn.functional.interpolate(scale_factor = 2, mode = 'nearest'),
			nn.Conv2d(in_channels = in_channels, out_channels = middle_channels, kernel_size = 3, padding = 1),
			nn.GroupNorm(num_gourps = middle_channels // 2, num_channels = middle_channels),
			nn.LeakyReLU(inplace = True),
			nn.Conv2d(in_channels = middle_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
			nn.GroupNorm(num_gourps = out_channels // 2, num_channels = out_channels),
			nn.LeakyReLU(inplace = True)
		)

	def forward(self, x):
		return self.block(x)

class SRDenseNetwork(nn.Module):
	def __init__(self, growth_rate = 32, block_config = (6, 12, 24, 16), num_init_features = 64, bot_neck = 4, drop_rate = 0, upscale_factor = 2):
		super(SRDenseNetwork, self).__init__()

		# First Convolution
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv2d(in_channels = 3, out_channels = num_init_features, kernel_size = 7, stride = 1, padding = 3, bias = False)),
			('g_norm0', nn.GroupNorm(num_groups = num_init_features // (growth_rate // 2) , num_channels = num_init_features)),
			('leaky_relu1', nn.LeakyReLU(inplace = True)),
			('avg_pool0', nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1)),
		]))

		# Dense Blocks
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(num_layers = num_layers, num_input_features = num_features, growth_rate = growth_rate, bot_neck = bot_neck, drop_rate = drop_rate)
			self.features.add_module('denseblock%d' % (i+1), block)
			num_features = num_features + num_layers * growth_rate
		
		# Final Normalization and Channel Bottleneck
		self.features.add_module('g_norm_f', nn.GroupNorm(num_groups = num_features // (growth_rate * bot_neck), num_channels = num_features))
		self.features.add_module('conv_bottleneck', nn.Conv2d(in_channel = num_features, out_channel = upscale_factor**2, kernel_size = 1, stride = 1, bias = False))

		# Upsample
		self.upsample_sr = nn.PixelShuffle(upscale_factor)

	def forward(self, x):
		features = self.features(x)
		out = F.leaky_relu(features, inplace = True)
		out = self.upsample_sr(out)
		return out

def densenetSR(**kwargs):
	
	model = SRDenseNetwork(growth_rate = 16, block_config = (3, 6, 12, 8), num_init_features = 64, bot_neck = 2, drop_rate = 0, upscale_factor = 2)

	return model
