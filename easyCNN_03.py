import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x


class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 60x60
            nn.ReLU(),
            nn.MaxPool2d(2),  # 30x30

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3)   # 5x5
        )

        self.channel_attention = ChannelAttention(64)
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class easyCNN_03(nn.Module):
    def __init__(self, use_metadata):
        super(easyCNN_03, self).__init__()
        self.use_metadata = use_metadata

        self.left_eye_net = EyeCNN()
        self.right_eye_net = EyeCNN()

        # if self.use_metadata:
        #     md = 3
        # else:
        #     md = 0
        md = 3 if self.use_metadata else 0

        # Expecting 3 additional inputs for head pose (yaw, pitch, roll)
        self.combined_fc = nn.Sequential(
            nn.Linear(256 * 2 + md, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # only lateral movement
        )

    def forward(self, left_eye, right_eye, head_pose=None):  # <-- new input
        left_feat = self.left_eye_net(left_eye)
        right_feat = self.right_eye_net(right_eye)

        if self.use_metadata:
            combined = torch.cat((left_feat, right_feat, head_pose), dim=1)  # <-- concatenate head pose
        else:
            combined = torch.cat((left_feat, right_feat), dim=1)
        output = self.combined_fc(combined)
        return output
