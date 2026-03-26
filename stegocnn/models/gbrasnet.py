import torch
import torch.nn as nn
from activation import TanH3
from utils import WeightExtractor


class PreProcessing(nn.Module):
    def __init__(
        self,
        srm_path,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
    ):
        super().__init__()

        if padding == "same":
            if isinstance(kernel_size, int):
                padding_val = kernel_size // 2
            else:
                padding_val = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding_val = padding

        self.conv_filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
            bias=True,
        )

        self.activation = TanH3()

        srm_weights, srm_bias = WeightExtractor.extract_srm_kernels(srm_path)

        with torch.no_grad():
            self.conv_filter.weight.copy_(srm_weights)
            self.conv_filter.bias.copy_(srm_bias)

        self.conv_filter.weight.requires_grad = False
        self.conv_filter.bias.requires_grad = False

        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            eps=1e-3,
            momentum=0.8,
            affine=True,
            track_running_stats=True,
        )

        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = True

        nn.init.ones_(self.conv_filter.bias)

    def forward(self, x):
        x = self.conv_filter(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


class FeatureExtractionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth_conv_kernel_size,
        separable_conv_kernel_size,
        depth_multiplier=3,
    ):
        super().__init__()

        if isinstance(depth_conv_kernel_size, int):
            depth_padding = depth_conv_kernel_size // 2
        else:
            depth_padding = (
                depth_conv_kernel_size[0] // 2,
                depth_conv_kernel_size[1] // 2,
            )

        self.depth_wise_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=depth_conv_kernel_size,
            groups=in_channels,
            padding=depth_padding,
        )
        self.separable_conv1 = SeparableConv(
            in_channels=in_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=separable_conv_kernel_size,
        )
        self.batch_norm1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True,
        )
        self.batch_norm1.weight.requires_grad = False
        self.batch_norm1.bias.requires_grad = True

        self.depth_wise_conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=depth_conv_kernel_size,
            groups=out_channels,
            padding=depth_padding,
        )
        self.separable_conv2 = SeparableConv(
            in_channels=out_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=separable_conv_kernel_size,
        )
        self.batch_norm2 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True,
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
    def __init__(
        self,
        in_channels,
        out_channels,
        depth_multiplier,
        kernel_size,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            padding_val = kernel_size // 2
        else:
            padding_val = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding_val,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels * depth_multiplier,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x


class DimensionalityReductionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        avg_kernel_size,
        avg_stride,
        conv_kernel_size,
        conv_stride,
        out_channels=60,
    ):
        super().__init__()
        self.average_pooling = nn.AvgPool2d(
            kernel_size=avg_kernel_size, stride=avg_stride
        )

        if isinstance(conv_kernel_size, int):
            padding_val = conv_kernel_size // 2
        else:
            padding_val = (conv_kernel_size[0] // 2, conv_kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=padding_val,
            bias=True,
        )
        self.activation = nn.ELU()

        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True,
        )

        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = True

    def forward(self, x):
        x = self.average_pooling(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


class SimpleConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True
    ):
        super().__init__()

        if isinstance(padding, str) and padding == "same":
            if isinstance(kernel_size, int):
                padding_val = kernel_size // 2
            else:
                padding_val = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding_val = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
            bias=bias,
        )
        self.activation = nn.ELU()
        self.batch_norm = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.8,
            eps=1e-3,
            affine=True,
            track_running_stats=True,
        )

        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.batch_norm.weight.requires_grad = False
        self.batch_norm.bias.requires_grad = True

    def forward(self, x):
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
    def __init__(
        self,
        srm_path,
        fe_stage1_in_channels=30,
        fe_stage1_out_channels=30,
        fe_stage1_depth_conv_kernel_size=(1, 1),
        fe_stage1_separable_conv_kernel_size=(3, 3),
        sc_stage1_in_channels=30,
        sc_stage1_out_channels=30,
        sc_stage1_kernel_size=(3, 3),
        sc_stage1_padding="same",
        sc_stage2_in_channels=30,
        sc_stage2_out_channels=30,
        sc_stage2_kernel_size=(3, 3),
        sc_stage2_padding="same",
        dr_stage1_in_channels=30,
        dr_stage1_out_channels=60,
        dr_stage1_avg_kernel_size=(2, 2),
        dr_stage1_avg_stride=(2, 2),
        dr_stage1_conv_kernel_size=(3, 3),
        dr_stage1_conv_stride=(1, 1),
        fe_stage2_in_channels=60,
        fe_stage2_out_channels=60,
        fe_stage2_depth_conv_kernel_size=(1, 1),
        fe_stage2_separable_conv_kernel_size=(3, 3),
        sc_stage3_in_channels=60,
        sc_stage3_out_channels=60,
        sc_stage3_kernel_size=(3, 3),
        sc_stage3_padding="same",
        dr_stage2_in_channels=60,
        dr_stage2_out_channels=60,
        dr_stage2_avg_kernel_size=(2, 2),
        dr_stage2_avg_stride=(2, 2),
        dr_stage2_conv_kernel_size=(3, 3),
        dr_stage2_conv_stride=(1, 1),
        dr_stage3_in_channels=60,
        dr_stage3_out_channels=60,
        dr_stage3_avg_kernel_size=(2, 2),
        dr_stage3_avg_stride=(2, 2),
        dr_stage3_conv_kernel_size=(3, 3),
        dr_stage3_conv_stride=(1, 1),
        dr_stage4_in_channels=60,
        dr_stage4_out_channels=30,
        dr_stage4_avg_kernel_size=(2, 2),
        dr_stage4_avg_stride=(2, 2),
        dr_stage4_conv_kernel_size=(1, 1),
        dr_stage4_conv_stride=(1, 1),
        sc_stage4_in_channels=30,
        sc_stage4_out_channels=2,
        sc_stage4_kernel_size=(1, 1),
        sc_stage4_padding="same",
        output_size=1,
    ):
        super().__init__()

        self.preprocessing = PreProcessing(
            srm_path=srm_path,
            in_channels=1,
            out_channels=30,
            kernel_size=(5, 5),
            padding="same",
        )

        # Feature Extracture Stage 1
        self.feature_extract1 = FeatureExtractionConv(
            in_channels=fe_stage1_in_channels,
            out_channels=fe_stage1_out_channels,
            depth_conv_kernel_size=fe_stage1_depth_conv_kernel_size,
            separable_conv_kernel_size=fe_stage1_separable_conv_kernel_size,
        )

        # Simple Convolutional Stage 1
        self.simple_conv1 = SimpleConv(
            in_channels=sc_stage1_in_channels,
            out_channels=sc_stage1_out_channels,
            kernel_size=sc_stage1_kernel_size,
            padding=sc_stage1_padding,
        )

        # Simple Convolutional Stage 2
        self.simple_conv2 = SimpleConv(
            in_channels=sc_stage2_in_channels,
            out_channels=sc_stage2_out_channels,
            kernel_size=sc_stage2_kernel_size,
            padding=sc_stage2_padding,
        )

        # Dimensionality Reduction Stage 1
        self.dim_reduc_1 = DimensionalityReductionConv(
            in_channels=dr_stage1_in_channels,
            out_channels=dr_stage1_out_channels,
            avg_kernel_size=dr_stage1_avg_kernel_size,
            avg_stride=dr_stage1_avg_stride,
            conv_kernel_size=dr_stage1_conv_kernel_size,
            conv_stride=dr_stage1_conv_stride,
        )

        # Feature Extracture Stage 2
        self.feature_extract2 = FeatureExtractionConv(
            in_channels=fe_stage2_in_channels,
            out_channels=fe_stage2_out_channels,
            depth_conv_kernel_size=fe_stage2_depth_conv_kernel_size,
            separable_conv_kernel_size=fe_stage2_separable_conv_kernel_size,
        )

        # Simple Convolutional Stage 3
        self.simple_conv3 = SimpleConv(
            in_channels=sc_stage3_in_channels,
            out_channels=sc_stage3_out_channels,
            kernel_size=sc_stage3_kernel_size,
            padding=sc_stage3_padding,
        )

        # Dimensionality Reduction Stage 2
        self.dim_reduc_2 = DimensionalityReductionConv(
            in_channels=dr_stage2_in_channels,
            out_channels=dr_stage2_out_channels,
            avg_kernel_size=dr_stage2_avg_kernel_size,
            avg_stride=dr_stage2_avg_stride,
            conv_kernel_size=dr_stage2_conv_kernel_size,
            conv_stride=dr_stage2_conv_stride,
        )

        # Dimensionality Reduction Stage 3
        self.dim_reduc_3 = DimensionalityReductionConv(
            in_channels=dr_stage3_in_channels,
            out_channels=dr_stage3_out_channels,
            avg_kernel_size=dr_stage3_avg_kernel_size,
            avg_stride=dr_stage3_avg_stride,
            conv_kernel_size=dr_stage3_conv_kernel_size,
            conv_stride=dr_stage3_conv_stride,
        )

        # Dimensionality Reduction Stage 4
        self.dim_reduc_4 = DimensionalityReductionConv(
            in_channels=dr_stage4_in_channels,
            out_channels=dr_stage4_out_channels,
            avg_kernel_size=dr_stage4_avg_kernel_size,
            avg_stride=dr_stage4_avg_stride,
            conv_kernel_size=dr_stage4_conv_kernel_size,
            conv_stride=dr_stage4_conv_stride,
        )

        # Simple Convolutional Stage 4
        self.simple_conv4 = SimpleConv(
            in_channels=sc_stage4_in_channels,
            out_channels=sc_stage4_out_channels,
            kernel_size=sc_stage4_kernel_size,
            padding=sc_stage4_padding,
        )

        # Output Stage
        self.output = OutputLayer(output_size=output_size)

    def forward(self, x):
        x = self.preprocessing(x)

        skip = self.feature_extract1(x)
        x = x + skip

        x = self.simple_conv1(x)
        x = self.simple_conv2(x)
        x = self.dim_reduc_1(x)
        skip = self.feature_extract2(x)
        x = x + skip

        x = self.simple_conv3(x)
        x = self.dim_reduc_2(x)
        x = self.dim_reduc_3(x)
        x = self.dim_reduc_4(x)
        x = self.simple_conv4(x)
        x = self.output(x)
        return x
