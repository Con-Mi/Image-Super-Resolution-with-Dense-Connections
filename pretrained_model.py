import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F

class PretrainedDenseSRmodel(nn.Module):
    def __init__(self, pretrained = False, upscale_factor = 2):
        super(PretrainedDenseSRmodel, self).__init__()
        
        self.encoder = models.densenet121(pretrained  = pretrained).features
        self.dense_layer1 = self.encoder[4]
        self.dense_layer2 = self.encoder[6]
        self.dense_layer3 = self.encoder[8]
        self.dense_layer4 = self.encoder[10]
        self.group_norm = nn.GroupNorm(num_groups = 1024 // 64, num_channels = 1024)
        self.bot_neck =  nn.Conv2d(in_channels = 1024, out_channels = 3*upscale_factor**2, kernel_size = 1, stride = 1, bias = False)
    
        self.upsample_sr = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        out = self.dense_layer1(x)
        out = self.dense_layer2(out)
        out = self.dense_layer3(out)
        out = self.dense_layer4(out)
        out = self.group_norm(out)
        out = self.bot_neck(out)
        out = F.selu(out)
        out = self.upsample_sr(out)

        return out

def pretrained_densenetSR(pretrained = False, upscale_factor = 2):
	
	model = PretrainedDenseSRmodel(pretrained = pretrained, upscale_factor=upscale_factor)

	return model
