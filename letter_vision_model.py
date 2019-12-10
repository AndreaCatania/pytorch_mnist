
import torch
import torch.nn as nn
import torch.nn.functional as F

class LetterVisionModel(nn.Module):
    def __init__(self):
        super(LetterVisionModel, self).__init__()

        self.layer_1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            padding_mode='zeros',
            bias=True,
        )

        self.layer_2 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.layer_3 = nn.Conv2d(
            in_channels = 16,
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            padding_mode='zeros',
            bias=True,
        )

        self.layer_4 = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.layer_flatten = nn.Flatten(start_dim = 1, end_dim = 3)

        self.layer_5 = nn.Linear(
            in_features = 7 * 7 * 8,
            out_features = 200,
            bias=True,
        )

        self.layer_6 = nn.Linear(
            in_features = 200,
            out_features = 10,
            bias=True,
        )

        self.softmax_6 = nn.Softmax(dim = 1)


    def forward(self, data):
        res = F.leaky_relu(self.layer_1(data))
        res = self.layer_2(res)
        res = F.leaky_relu(self.layer_3(res))
        res = self.layer_4(res)
        res = self.layer_flatten(res)
        res = F.leaky_relu(self.layer_5(res))
        res = self.softmax_6(self.layer_6(res))
        return res
    
