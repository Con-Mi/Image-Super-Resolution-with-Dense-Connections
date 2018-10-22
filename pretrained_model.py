import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F

class PretrainedDenseSRmodel(nn.Module):
    def __init__(self, pretrained = False, upscale_factor = 2):
        super(PretrainedDenseSRmodel, self).__init__()
        
        self.encoder = models.densenet121(pretrained  = pretrained).features
        self.low_conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.bn = self.encoder[1]
        self.relu = self.encoder[2]
        self.dense_layer1 = self.encoder[4]
        self.bot_neck1 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 1, stride = 1, bias = False)

        self.dense_layer2 = self.encoder[6]
        self.bot_neck2 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1, stride = 1, bias = False)

        self.dense_layer3 = self.encoder[8]
        self.bot_neck3 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1, stride = 1, bias = False)

        # self.dense_layer4 = self.encoder[10]
        
        self.group_norm = nn.GroupNorm(num_groups = 4, num_channels = 512)
        self.bot_neck =  nn.Conv2d(in_channels = 512, out_channels = 3*upscale_factor**2, kernel_size = 1, stride = 1, bias = False)
    
        self.upsample_sr = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        out = self.low_conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.dense_layer1(out)
        out = self.bot_neck1(out)

        out = self.dense_layer2(out)
        out = self.bot_neck2(out)

        out = self.dense_layer3(out)
        out = self.bot_neck3(out)

        #out = self.dense_layer4(out)

        out = self.group_norm(out)
        out = F.elu(out)
        out = self.bot_neck(out)
        out = F.elu(out)
        out = self.upsample_sr(out)

        return out

def pretrained_densenetSR(pretrained = False, upscale_factor = 2):
	
	model = PretrainedDenseSRmodel(pretrained = pretrained, upscale_factor=upscale_factor)

	return model
