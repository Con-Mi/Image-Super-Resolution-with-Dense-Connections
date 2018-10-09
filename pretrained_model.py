import torch
from torchvision import models
from torch import nn

class PretrainedDenseSRmodel(nn.Module):
    def __init__(self, pretrained = False):
        super(PretrainedSRmodel, self).__init__()
        
        self.encoder = model.densenet121(pretrained  = pretrained)
        self.dense_layer1 = self.encoder[4]
        self.dense_layer2 = self.encoder[6]
        self.dense_layer3 = self.encoder[8]
        self.dense_layer4 = self.encoder[10]
        
        self.classifier = nn.PixelShuffle()
        
    def forward(self, x):
        out = self.dense_layer1(x)
        out = self.dense_layer2(out)
        out = self.dense_layer3(out)
        out = self.dense_layer4(out)
        return self.classifier(out)
