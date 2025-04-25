import matplotlib.pyplot as plt
from torch import nn
import torch


LATENT_CHANNELS = 16

class easyCNN(nn.Module):

    def __init__(self, use_metadata):
        super().__init__()

        self.use_metadata = use_metadata

        self.norm = nn.GroupNorm(num_groups=1, num_channels=2, affine=False)

        self.conv0 = nn.Sequential(
            # 2@60x60 -> 2@58x58
            nn.Conv2d(in_channels=2, out_channels=LATENT_CHANNELS, kernel_size=(3, 3),  stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            # nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            # 2@58x58 -> 2@56x56
            nn.Conv2d(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            # nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            # 2@56x56 -> 2@54x54
            nn.Conv2d(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            # 16@54x54 -> 16@52x52
            nn.Conv2d(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, kernel_size=(3, 3), stride=(1, 1),
                      padding=(0, 0), groups=1),
            nn.BatchNorm2d(LATENT_CHANNELS),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        # 16@11x11-> 16*11*11 = 141376
        self.flatten0 = nn.Flatten()

        # 141376 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=LATENT_CHANNELS*52*52, out_features=128),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )

        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )

        # 128 -> 128
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.5),
            nn.Tanh()
        )

        # 128 -> 72
        self.linear3 = nn.Linear(
            in_features=128, out_features=1
        )

        if self.use_metadata:
            self.FiLM0 = nn.Sequential(
                nn.Linear(in_features=4, out_features=2*128),
                nn.Dropout(p=0.5),
                nn.Tanh()
            )
            self.FiLM1 = nn.Sequential(
                nn.Linear(in_features=2*128, out_features=128),
                nn.Dropout(p=0.5),
                nn.Tanh()
            )
            self.FiLM2 = nn.Sequential(
                nn.Linear(in_features=128, out_features=2*128),
                nn.Dropout(p=0.5),
                nn.Tanh()
            )

        self.GRU = nn.GRU(input_size=128, hidden_size=128,
               num_layers=3, batch_first=True,
               dropout=0.5,
               bidirectional=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_left, image_right, metadata):

        x = torch.cat((image_left, image_right), dim=1)

        global_max = torch.max(torch.max(torch.max(x)))
        global_min = torch.min(torch.min(torch.min(x)))

        # normalisation [0,1]
        x -= global_min
        x /= (global_max - global_min)
        # squaring for contrast enhancement
        x *= x

        # Normalise batch
        # x = self.norm(input_data)

        # plt.imshow(x[0, 0, :, :].cpu().detach().numpy())
        # plt.show()

        # Neural Net
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten0(x)

        x = self.linear0(x)

        if self.use_metadata:
            # calculate FiLM layers
            m = self.FiLM0(metadata)
            m = self.FiLM1(m)
            m = self.FiLM2(m)
            alpha = m[:, :128]
            beta = m[:, 128:]
            # conduct FiLM
            x = alpha * x + beta

        x = self.linear1(x)
        x = self.linear2(x)

        (x, _) = self.GRU(x)

        predictions = self.linear3(x)

        # predictions = self.softmax(predictions)

        return predictions

