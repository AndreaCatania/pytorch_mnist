import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloModel(nn.Module):
    def __init__(self):
        super(YoloModel, super).__init__()
    
        self.leaky_relu = nn.LeakyReLu()

        self.l1_conv = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding=3,
            padding_mode = 'circular',
            bias = True,
        )
        self.l1_batch_norm = nn.BatchNorm2d(64)
        self.l1_mp = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )

        self.l2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 192,
            kernel_size = 3,
            stride = 1,
            padding=1,
            padding_mode = 'circular',
            bias = True,
        )
        self.l2_batch_norm = nn.BatchNorm2d(192)
        self.layer_2_mp = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
        )

        self.l3 = nn.Conv2d(
            in_channels = 192,
            out_channels = 128,
            kernel_size = 1,
            stride = 1,
            bias = True,
        )
        self.l3_batch_norm = nn.BatchNorm2d(128)

        self.l4 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            padding_mode='circular',
            bias = True,
        )
        self.l4_batch_norm = nn.BatchNorm2d(256)

        self.l5 = nn.Conv2d(
            in_channels = 256,
            out_channels = 256,
            kernel_size = 1,
            stride = 1,
            bias = True,
        )
        self.l5_batch_norm = nn.BatchNorm2d(256)





    def forward(self, inp):
        pass
