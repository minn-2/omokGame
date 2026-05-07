import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOModel(nn.Module):
    def __init__(self, board_size=15):
        """
        Actor-Critic 구조
        Actor:  어디에 둘지 결정 (착수 확률)
        Critic: 현재 상황의 승률 예측
        """
        super(PPOModel, self).__init__()
        self.board_size   = board_size
        self.flatten_size = 128 * board_size * board_size

        # CNN: 보드 패턴 인식
        self.conv_block = nn.Sequential(
            # 1층: 국소 패턴 인식 (3x3 영역)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 2층: 복잡한 패턴 인식 (3목, 4목)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 3층: 고차원 전술 특징 추출
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Actor: 착수 확률 계산
        self.actor = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size)
        )

        # Critic: 승률 예측 (-1 ~ 1)
        self.critic = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # CNN 특징 추출
        x      = self.conv_block(x)
        # Flatten (Batch, C, H, W) → (Batch, Features)
        x      = x.view(x.size(0), -1)
        # Actor: 확률 분포
        probs  = F.softmax(self.actor(x), dim=-1)
        # Critic: 가치 점수
        value  = self.critic(x)
        return probs, value