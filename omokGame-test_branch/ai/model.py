import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOModel(nn.Module):

    def __init__(self, board_size=15):

        super(PPOModel, self).__init__()

        self.board_size = board_size

        # CNN 특징 추출
        self.conv_block = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # Flatten 크기
        self.flatten_size = (
            128
            * board_size
            * board_size
        )

        # Actor
        self.actor = nn.Sequential(

            nn.Linear(
                self.flatten_size,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                board_size * board_size
            )
        )

        # Critic
        self.critic = nn.Sequential(

            nn.Linear(
                self.flatten_size,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                1
            )
        )

    def forward(self, x):

        # CNN 통과
        x = self.conv_block(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Actor
        logits = self.actor(x)

        probs = F.softmax(
            logits,
            dim=-1
        )

        # Critic
        value = self.critic(x)

        return probs, value
