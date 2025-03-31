
from torch import nn


class easyCNN_01(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=1, num_channels=1)

        self.conv0 = nn.Sequential(
            # 1@100x200 -> 1@98x198
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3),  stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            # 1@98x198 -> 1@96x196
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            # 1@96x196 -> 1@94x194
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0.5),
            nn.LeakyReLU()
        )

        # 5@1x16-> 5*1*16 = 80
        self.flatten0 = nn.Flatten()

        # 80 -> 128
        self.linear0 = nn.Sequential(
            nn.Linear(in_features=94*194, out_features=1024),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        # 128 -> 128
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
        # 128 -> 72
        self.linear2 = nn.Linear(
            in_features=512, out_features=1
        )

        self.FiLM0 = nn.Sequential(
            nn.Linear(in_features=4, out_features=2*128),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        self.FiLM1 = nn.Sequential(
            nn.Linear(in_features=2*128, out_features=2 * 256),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )
        self.FiLM2 = nn.Sequential(
            nn.Linear(in_features=2 * 256, out_features=2 * 1024),
            nn.Dropout(p=0.5),
            nn.LeakyReLU()
        )

        self.act = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data, metadata):

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
        alpha = m[:, :1024]
        beta = m[:, 1024:]

        # conduct FiLM
        x = alpha * x + beta

        x = self.linear1(x)
        predictions = self.linear2(x)

        # predictions = self.softmax(predictions)

        return predictions

