import torch
import torch.nn as nn

# --------------------------
# Attention Modules
# --------------------------
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

# --------------------------
# Residual Block
# --------------------------
class EyeResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EyeResNetBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res_block(x) + self.skip(x))

# --------------------------
# Attention-Enhanced EyeCNN
# --------------------------
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.block1 = EyeResNetBlock(1, 32)      # 60x60 → 30x30
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = EyeResNetBlock(32, 64)     # 30x30 → 15x15
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = EyeResNetBlock(64, 128)    # 15x15 → 7x7
        self.pool3 = nn.MaxPool2d(2)

        self.channel_attention = ChannelAttention(128)
        self.spatial_attention = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        x = self.fc(x)
        return x

# --------------------------
# Final Temporal Model with Attention
# --------------------------
class easyCNN_04(nn.Module):
    def __init__(self, eye_feature_dim=256, lstm_hidden_dim=128, output_dim=1):
        super(easyCNN_04, self).__init__()

        self.left_eye_cnn = EyeCNN()
        self.right_eye_cnn = EyeCNN()

        # LSTM input: concatenated eye features
        self.lstm = nn.LSTM(input_size=eye_feature_dim * 2,
                            hidden_size=lstm_hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Gaze direction: (x, y) or (yaw, pitch)
        )

    def forward(self, left_seq, right_seq):
        """
        Inputs:
            left_seq:  (batch, time, 1, 60, 60)
            right_seq: (batch, time, 1, 60, 60)
        Output:
            gaze_pred: (batch, 2)
        """
        B, T, C, H, W = left_seq.shape

        # Flatten time dimension for CNN processing
        left_seq = left_seq.view(B * T, C, H, W)
        right_seq = right_seq.view(B * T, C, H, W)

        # Extract per-frame eye features
        left_feat = self.left_eye_cnn(left_seq)   # (B*T, 256)
        right_feat = self.right_eye_cnn(right_seq)

        # Reconstruct temporal sequence
        combined_feat = torch.cat([left_feat, right_feat], dim=1)  # (B*T, 512)
        combined_feat = combined_feat.view(B, T, -1)  # (B, T, 512)

        # LSTM over time
        lstm_out, _ = self.lstm(combined_feat)  # (B, T, hidden)

        # Predict only at final time step
        gaze_pred = self.fc(lstm_out[:, -1, :])  # (B, 1)

        return gaze_pred