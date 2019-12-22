import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloModel(nn.Module):
    def __init__(self):
        super(YoloModel, self).__init__()
    
        self.leaky_relu = nn.LeakyReLU()

        self.l1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding=3,
            padding_mode = 'circular',
            bias = True,
        )
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
        self.l2_mp = nn.MaxPool2d(
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

        self.l6 = nn.Conv2d(
            in_channels = 256,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            padding_mode='circular',
            bias = True,
        )
        self.l6_batch_norm = nn.BatchNorm2d(512)

        self.l7 = nn.Conv2d(
            in_channels = 512,
            out_channels = 256,
            kernel_size = 1,
            stride = 1,
            bias = True,
        )
        self.l7_batch_norm = nn.BatchNorm2d(256)

        self.l8 = nn.Conv2d(
            in_channels = 256,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            padding_mode='circular',
            bias = True,
        )
        self.l8_batch_norm = nn.BatchNorm2d(512)

        self.l_flatten = nn.Flatten(
            start_dim = 1,
            # end_dim = 3,
        )

        self.dropout = nn.Dropout(
            p = 0.5,
        )

        self.l9 = nn.Linear(
            in_features = 4608,
            out_features = 2304,
            bias=True,
        )

        quads = 4
        classes = 10

        self.l10 = nn.Linear(
            in_features = 2304,
            out_features = quads * (1 + 2 + 2 + classes), # 60
            bias=True,
        )


    def forward(self, inp):
        d = self.l1(inp)
        d = self.leaky_relu(d)
        d = self.l1_mp(d)

        d = self.l2(d)
        d = self.l2_batch_norm(d)
        d = self.leaky_relu(d)
        d = self.l2_mp(d)

        d = self.l3(d)
        d = self.l3_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l4(d)
        d = self.l4_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l5(d)
        d = self.l5_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l6(d)
        d = self.l6_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l7(d)
        d = self.l7_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l8(d)
        d = self.l8_batch_norm(d)
        d = self.leaky_relu(d)

        d = self.l_flatten(d)

        d = self.dropout(d)

        d = self.l9(d)
        d = self.leaky_relu(d)

        d = self.l10(d)
        # Linear activation

        return d
    

    """ Returns the quads count """
    def yolo_quads(self):
        return 4


    """ Returns the quads of this model """
    def yolo_data(self, image_width, image_height):
        quads = []
        row_q_count = 2
        cell_q_count = 2
        w = image_width / row_q_count
        h = image_height / cell_q_count
        for row in range(row_q_count):
            for cell in range(cell_q_count):
                quads.append([cell * w, row * h])
        return quads, w, h
