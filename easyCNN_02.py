import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeCNN(nn.Module):
    def __init__(self):

        super(EyeCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 30, 30]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 30, 30]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 15, 15]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 15, 15]
            nn.ReLU(),
            nn.MaxPool2d(2)  # [B, 128, 7, 7]
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 512)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class easyCNN_02(nn.Module):
    def __init__(self):
        super(easyCNN_02, self).__init__()
        self.left_eye_net = EyeCNN()
        self.right_eye_net = EyeCNN()

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output: 1D gaze vector (radian)
        )

    def forward(self, left_eye, right_eye):
        left_feat = self.left_eye_net(left_eye)
        right_feat = self.right_eye_net(right_eye)

        combined = torch.cat((left_feat, right_feat), dim=1)
        output = self.fc(combined)
        return output