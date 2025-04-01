
from torch import nn
import torch


LATENT_CHANNELS = 16

class easyCNN_01(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=1, num_channels=2)

        self.conv0 = nn.Sequential(
            # 2@100x100 -> 16@98x98
            nn.Conv2d(in_channels=2, out_channels=LATENT_CHANNELS, kernel_size=(3, 3),  stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            # 16@98x98 -> 16@96x96
            nn.Conv2d(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            # 16@96x96 -> 16@94x94
            nn.Conv2d(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(LATENT_CHANNELS),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        # 16@94x94-> 16*94*94 = 141376
        self.flatten0 = nn.Flatten()

        # 141376 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=LATENT_CHANNELS*94*94, out_features=128),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        # 128 -> 72
        self.linear2 = nn.Linear(
            in_features=128, out_features=1
        )

        self.FiLM0 = nn.Sequential(
            nn.Linear(in_features=4, out_features=2*128),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        self.FiLM1 = nn.Sequential(
            nn.Linear(in_features=2*128, out_features=2*128),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        self.FiLM2 = nn.Sequential(
            nn.Linear(in_features=2*128, out_features=2*128),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )

        # 512 -> 512
        self.act = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_left, image_right, metadata):

        input_data = torch.cat((image_left, image_right), dim=1)

        # Normalise batch
        x = self.norm(input_data)

        # Neural Net
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten0(x)

        x = self.linear0(x)

        # calculate FiLM layers
        m = self.FiLM0(metadata)
        m = self.FiLM1(m)
        m = self.FiLM2(m)
        alpha = m[:, :128]
        beta = m[:, 128:]

        # conduct FiLM
        x = alpha * x + beta

        x = self.linear1(x)
        predictions = self.linear2(x)

        # predictions = self.softmax(predictions)

        return predictions

