import torch
import torch.nn as nn
from activation import TanH3

class PreProcessing(nn.Module):
    
    def __init__(self, srm_weights):
        super().__init__()
        self.conv_filter = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_filter.weight = nn.Parameter(torch.tensor(srm_weights), requires_grad=False)
        nn.init.ones(self.conv_filter.bias)
        self.activation = TanH3()
        self.batch_norm = nn.BatchNorm2d(30, eps=1e-3, momentum=.2, affine=True)
        
    def forward(self, x):
        x = self.conv_filter(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x



        
        