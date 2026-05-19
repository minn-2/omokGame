import numpy as np
import torch
import copy

from ai.engine import Engine
from ai.agent import PPOAgent


# 설정
BOARD_SIZE = 15
NUM_EPISODES = 10000

PLAYER_1 = 1
PLAYER_2 = 2


def train():

    # 엔진 생성
    engine = Engine(BOARD_SIZE)

    # 학습 AI
    agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_1
    )

    # 상대 AI (초기에는 자기 자신 복사)
    opponent_agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_2
    )

    opponent_agent.policy.load_state_dict(
        copy.deepcopy(
            agent.policy.state_dict()
        )
    )

    wins = 0
    losses = 0
    draws = 0

    best_win_rate = 0.0

    print('Self-Play PPO 학습 시작')
    print('Device:', agent.device)
    print('-' * 50)

    for episode in range(NUM_EPISODES):

        engine.reset()

        while not engine.is_over:

            # 학습 AI 차례
            if engine.current_player == PLAYER_1:

                move = agent.decide_next_move(
                    engine
                )

                if move is None:
                    break

                success = engine.make_move(
                    *move
                )

                if not success:
                    break

                reward = agent.calculate_reward(
                    engine
                )

                agent.store_reward(reward)

            # 상대 AI 차례
            else:

                move = opponent_agent.decide_next_move(
                    engine
                )

                if move is None:
                    break

                success = engine.make_move(
                    *move
                )

                if not success:
                    # 실패 패널티 저장
                    agent.store_reward(-10.0)

                    break

        # 결과 기록
        if engine.winner == PLAYER_1:

            wins += 1

        elif engine.winner == 0:

            draws += 1

        else:

            losses += 1

        # PPO 업데이트
        agent.update()

        # 출력
        if (episode + 1) % 100 == 0:

            total = episode + 1

            win_rate = (
                wins / total
            ) * 100

            print(
                'Episode {}/{} | '
                '승 {} | '
                '패 {} | '
                '무 {} | '
                '승률 {:.1f}%'
                .format(
                    total,
                    NUM_EPISODES,
                    wins,
                    losses,
                    draws,
                    win_rate
                )
            )

            # 최고 승률 갱신 시 저장
            if win_rate > best_win_rate:

                best_win_rate = win_rate

                print(
                    '최고 승률 갱신 → 모델 저장'
                )

                agent.save()

                # 상대 모델 갱신
                opponent_agent.policy.load_state_dict(
                    copy.deepcopy(
                        agent.policy.state_dict()
                    )
                )

                opponent_agent.policy_old.load_state_dict(
                    opponent_agent.policy.state_dict()
                )

                print(
                    'Self-Play 상대 모델 갱신 완료'
                )

    print('-' * 50)

    print('학습 완료')

    print(
        '최종 승률: {:.1f}%'
        .format(
            (wins / NUM_EPISODES) * 100
        )
    )

    agent.save()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    train()
