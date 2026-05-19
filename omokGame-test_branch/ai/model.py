# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class PPOModel(nn.Module):

    def __init__(self, board_size=15, in_channels=1):

        super().__init__()

        self.board_size = board_size

        NUM_FILTERS = 128

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, NUM_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(NUM_FILTERS),
            ResidualBlock(NUM_FILTERS),
            ResidualBlock(NUM_FILTERS),
            ResidualBlock(NUM_FILTERS),
        )

        self.flatten_size = NUM_FILTERS * board_size * board_size

        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, board_size * board_size),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = torch.flatten(x, 1)

        logits = self.actor(x)
        probs  = F.softmax(logits, dim=-1)
        value  = self.critic(x)

        return probs, value