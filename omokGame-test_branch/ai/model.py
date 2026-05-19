import torch
import torch.nn as nn
import torch.nn.functional as F

# 잔차 블록
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x, inplace=True)


# PPO 모델 (Actor-Critic)
class PPOModel(nn.Module):
    # 기본 하이퍼파라미터
    CHANNELS   = 64   # 백본 채널 수
    RES_BLOCKS = 5    # 잔차 블록 개수

    def __init__(self, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        n = board_size * board_size
        C = self.CHANNELS

        # 공유 백본 (Shared Backbone)
        # 오목판의 국소 패턴(3·4목 형성, 금수 등)을 계층적으로 추출
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            *[ResBlock(C) for _ in range(self.RES_BLOCKS)]
        )

        # Actor 헤드 (정책 결정기)
        # 각 칸에 돌을 놓을 확률 계산
        self.actor_head = nn.Sequential(
            nn.Conv2d(C, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * n, n),
        )

        # Critic 헤드 (가치 평가기)
        # 현재 보드 상황을 -1(패배) ~ +1(승리) 사이의 스칼라로 평가
        self.critic_head = nn.Sequential(
            nn.Conv2d(C, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(n, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feat  = self.backbone(self.stem(x))    # 공유 특징 추출
        logits = self.actor_head(feat)          # (B, N)
        probs  = F.softmax(logits, dim=-1)      # 합이 1이 되도록 정규화
        value  = self.critic_head(feat)         # (B, 1)
        return probs, value