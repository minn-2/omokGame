import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOModel(nn.Module):

    def __init__(self, board_size=15):
        """
        PPO Actor-Critic 모델

        board_size:
            오목판 크기
        """

        super(PPOModel, self).__init__()

        self.board_size = board_size

        # CNN 특징 추출 블록
        # 오목판의 패턴(3목, 4목, 연결 구조 등)을
        # 학습하기 위한 CNN 계층

        self.conv_block = nn.Sequential(

            # Conv Layer 1
            # 입력:
            #   1채널 바둑판
            # 출력:
            #   64개 특징맵

            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            # Conv Layer 2
            # 64채널 → 128채널

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            # Conv Layer 3
            # 128채널 유지
            # 더 복잡한 패턴 학습

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # Flatten 크기 계산
        # CNN 출력:
        #   (128, board_size, board_size)
        # 이를 1차원 벡터로 변환

        self.flatten_size = ( 128  * board_size * board_size )

        # Actor Network
        # 다음 착수 위치 선택
        self.actor = nn.Sequential(
            nn.Linear( self.flatten_size, 256 ), nn.ReLU(), nn.Linear( 256, board_size * board_size )
        )

        # Critic Network
        # 현재 상태 가치 평가
        self.critic = nn.Sequential(
            nn.Linear( self.flatten_size, 256 ), nn.ReLU(), nn.Linear( 256, 1 )
        )

    def forward(self, x):

        # CNN 특징 추출
        x = self.conv_block(x)

        # Flatten
        # CNN 출력 텐서를
        # 1차원 벡터로 변환
        x = torch.flatten(x, 1)

        # Actor 계산
        # 각 위치별 점수 계산
        logits = self.actor(x)

        # Softmax 안정성 향상
        logits = ( logits - logits.max( dim=-1, keepdim=True )[0] )

        # 확률 변환
        probs = F.softmax( logits, dim=-1 )

        # Critic 계산
        # 현재 상태 가치 평가
        value = self.critic(x)

        return probs, value
