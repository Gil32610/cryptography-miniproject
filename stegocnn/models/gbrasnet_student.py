import torch
import torch.nn as nn
from activation import TanH3
from utils import WeightExtractor

class PreProcessing(nn.Module):
    def __init__(self,
                 srm_path,
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
      
        nn.init.ones_(self.conv_filter.bias)
        self.activation = TanH3()
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=.8,
            affine=True,
            track_running_stats=True
            )
        
        srm_weights, srm_bias = WeightExtractor.extract_srm_kernels(srm_path)
    
        with torch.no_grad():
            self.conv_filter.weight.copy_(srm_weights)
            self.conv_filter.bias.copy_(srm_bias)
            
        self.conv_filter.weight.requires_grad = False
        self.conv_filter.bias.requires_grad = False
        
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
            num_features=out_channels,
            momentum=.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True
        )
        self.batch_norm2.weight.requires_grad = False
        self.batch_norm2.bias.requires_grad = True
        
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
            padding=1,
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

class DimensionalityReductionConv(nn.Module):
    def __init__(self,
                 in_channels,
                 avg_kernel_size,
                 avg_stride,
                 conv_kernel_size,
                 conv_stride,
                 out_channels=60,
                 ):
        super().__init__()
        self.average_pooling = nn.AvgPool2d(
            kernel_size=avg_kernel_size,
            stride=avg_stride
            )
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride
            )
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True
        )
        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = True
    
    def forward(self, x):
        x = self.average_pooling(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return x

class SimpleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 bias = True
                 ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.activation = nn.ELU()
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True
        )
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            
    def forward(self,x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x
    
class OutputLayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x = x.view(x.size(0), -1)
        return x
    
class GBRASNET(nn.Module):
    def __init__(self, srm_path):
        super().__init__()
        
        # Preprocessing Stage
        self.preprocessing = PreProcessing(srm_path=srm_path,in_channels=1, out_channels=30, kernel_size=(5,5),padding=2)

        # Feature Extracture Stage 1
        self.feature_extract1 = FeatureExtractionConv(in_channels=30, out_channels=30, depth_conv_kernel_size=(1,1), separable_conv_kernel_size=(3,3))

        # Simple Convolutional Stage 1
        self.simple_conv1 = SimpleConv(in_channels=30, out_channels=30, kernel_size=(3,3))

        # Simple Convolutional Stage 2
        self.simple_conv2 = SimpleConv(in_channels=30, out_channels=30, kernel_size=(3,3))

        # Dimensionality Reduction Stage 1
        self.dim_reduc_1 = DimensionalityReductionConv(in_channels=30, avg_kernel_size=(2,2), avg_stride=(2,2), conv_kernel_size=(3,3), conv_stride=(1,1))

        # Feature Extracture Stage 2
        self.feature_extract2 = FeatureExtractionConv(in_channels=60, out_channels=60, depth_conv_kernel_size=(1,1), separable_conv_kernel_size=(3,3))

        # Simple Convolutional Stage 3
        self.simple_conv3 = SimpleConv(in_channels=60, out_channels=60, kernel_size=(3,3))

        # Dimensionality Reduction Stage 2
        self.dim_reduc_2 = DimensionalityReductionConv(in_channels=60, avg_kernel_size=(2,2), avg_stride=(2,2), conv_kernel_size=(3,3), conv_stride=(1,1))

        # Dimensionality Reduction Stage 3
        self.dim_reduc_3 = DimensionalityReductionConv(in_channels=60, avg_kernel_size=(2,2), avg_stride=(2,2), conv_kernel_size=(3,3), conv_stride=(1,1))

        # Dimensionality Reduction Stage 4
        self.dim_reduc_4 = DimensionalityReductionConv(in_channels=60,out_channels=30, avg_kernel_size=(2,2), avg_stride=(2,2), conv_kernel_size=(1,1), conv_stride=(1,1))

        # Simple Convolutional Stage 4
        self.simple_conv4 = SimpleConv(in_channels=30, out_channels=2, kernel_size=(1,1), padding=0)

        # Output Stage
        self.output = OutputLayer(output_size=1)

    def forward(self, x):
        x = self.preprocessing(x)
        
        skip = self.feature_extract1(x)
        x += skip

        x = self.simple_conv1(x)
        x = self.dim_reduc_1(x)
        skip = self.feature_extract2(x)
        x += skip

        x = self.simple_conv3(x)
        x = self.dim_reduc_2(x)
        x = self.dim_reduc_3(x)
        x = self.dim_reduc_4(x)
        x = self.simple_conv4(x)
        x = self.output(x)
        return x