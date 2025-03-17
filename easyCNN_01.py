
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt

class easyCNN_01(nn.Module):

    def __init__(self, num_output_classes=1):
        super().__init__()

        self.num_classes = num_output_classes

        self.norm = nn.GroupNorm(num_groups=1, num_channels=5)

        self.conv0 = nn.Sequential(
            # 5@5x28 -> 5@3x24
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 5),  stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            # 5@3x24 -> 5@1x20
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 5), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            # 5@1x20 -> 5@1x16
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        # 5@1x16-> 5*1*16 = 80
        self.flatten0 = nn.Flatten()

        # 80 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=80, out_features=128),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.8),
            nn.LeakyReLU()
        )
        # 128 -> 72
        self.linear2 = nn.Linear(
            in_features=128, out_features=self.num_classes
        )

        # 512 -> 512
        self.act = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):

        # print(input_data.shape)

        # Normalise batch
        x = self.norm(input_data)

        # Neural Net
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten0(x)

        x = self.linear0(x)
        x = self.linear1(x)
        predictions = self.linear2(x)

        predictions = self.softmax(predictions)

        return predictions

