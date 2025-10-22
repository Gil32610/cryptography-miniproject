import torch
import torch.nn as nn
from activation import TanH3

class PreProcessing(nn.Module):
    def __init__(self,
                 srm_weights,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding='same',
                 ):
        super().__init__()
        self.conv_filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
            )
        self.conv_filter.weight = nn.Parameter(
            torch.tensor(srm_weights),
            requires_grad=False
            )
        nn.init.ones(self.conv_filter.bias)
        self.activation = TanH3()
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=.8,
            affine=True,
            track_running_stats=True
            )
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = True
        
    def forward(self, x):
        x = self.conv_filter(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


class FeatureExtractionConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 depth_conv_kernel_size,
                 separable_conv_kernel_size,
                 depth_multiplier=3
                 ):
        super().__init__()
        self.depth_wise_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=depth_conv_kernel_size,
            groups=in_channels
        )
        self.separable_conv1 = SeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=separable_conv_kernel_size
            )
        self.batch_norm1 = nn.BatchNorm2d(
            momentum=.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True
        )
        self.batch_norm1.weight.requires_grad = False
        self.batch_norm1.bias.requires_grad = True
        
        
        self.depth_wise_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=depth_conv_kernel_size,
            groups=in_channels
        )
        self.separable_conv2 = SeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=separable_conv_kernel_size
            )
        self.batch_norm2 = nn.BatchNorm2d(
            momentum=.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True
        )
        
    def forward(self, x):
        x = self.depth_wise_conv1(x)
        x = self.separable_conv1(x)
        x = self.batch_norm1(x)
        x = self.depth_wise_conv2(x)
        x = self.separable_conv2(x)
        x = self.batch_norm2(x)
        return x
    
class SeparableConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 depth_multiplier,
                 kernel_size,
                 ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False
            )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels*depth_multiplier,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )
        self.activation = nn.ELU()
        
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x

        
        