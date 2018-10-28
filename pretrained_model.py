import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F
from torch import nn

class BottleNeck(nn.Module):
    def __init__(self, in_chnl, out_chnl):
        super(BottleNeck, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=out_chnl)
        self.elu = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.gn(out)
        out = self.elu(out)

class PretrainedDenseSRmodel(nn.Module):
    def __init__(self, pretrained = False, upscale_factor = 2):
        super(PretrainedDenseSRmodel, self).__init__()
        
        self.encoder = models.densenet121(pretrained  = pretrained).features
        self.low_conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=64)
        self.elu = nn.ELU(inplace=True)
        
        self.dense_layer1 = self.encoder[4]
        self.bot_neck1 = BottleNeck(in_chnl=256, out_chnl=128)
        
        self.dense_layer2 = self.encoder[6]
        self.bot_neck2 = BottleNeck(in_chnl=512, out_chnl=256)
        
        self.dense_layer3 = self.encoder[8]
        self.bot_neck3 = BottleNeck(in_chnl=1024, out_chnl=512)
        
        self.dense_layer4 = self.encoder[10]
        
        self.group_norm = nn.GroupNorm(num_groups = 16, num_channels = 256)
        self.bot_neck = BottleNeck(in_chnl=256, out_chnl=3*upscale_factor**2)
        self.upsample_sr = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        out = self.low_conv(x)
        out = self.gn(out)
        out = self.elu(out)

        out = self.dense_layer1(out)
        out = self.bot_neck1(out)
        """
        out = self.dense_layer2(out)
        out = self.bot_neck2(out)

        out = self.dense_layer3(out)
        out = self.bot_neck3(out)

        out = self.dense_layer4(out)

        out = self.group_norm(out)
        out = F.elu(out)
        out = self.bot_neck(out)
        out = self.upsample_sr(out)
        """
        return out

def pretrained_densenetSR(pretrained = False, upscale_factor = 2):
	model = PretrainedDenseSRmodel(pretrained = pretrained, upscale_factor=upscale_factor)
	return model
