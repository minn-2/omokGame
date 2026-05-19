import numpy as np
import torch
import copy

from ai.engine import Engine
from ai.agent import PPOAgent


BOARD_SIZE = 15
NUM_EPISODES = 500000
LOG_INTERVAL = 100

PLAYER_1 = 1
PLAYER_2 = 2


def train():

    engine = Engine(BOARD_SIZE)

    agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_1,
        in_channels=3
    )

    opponent_agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_2,
        in_channels=3
    )

    opponent_agent.policy.load_state_dict(
        copy.deepcopy(agent.policy.state_dict())
    )
    opponent_agent.policy_old.load_state_dict(
        copy.deepcopy(agent.policy.state_dict())
    )

    total_wins   = 0
    total_losses = 0
    total_draws  = 0

    interval_wins   = 0
    interval_losses = 0
    interval_draws  = 0

    best_win_rate = 0.0

    print('Self-Play PPO 학습 시작')
    print('Device:', agent.device)
    print('-' * 50)

    for episode in range(NUM_EPISODES):

        engine.reset()
        abnormal_end = False

        while not engine.is_over:

            if engine.current_player == PLAYER_1:

                move = agent.decide_next_move(engine)

                if move is None:
                    abnormal_end = True
                    break

                success = engine.make_move(*move)

                if not success:
                    agent.store_reward(-10.0)
                    abnormal_end = True
                    break

                reward = agent.calculate_reward(engine)
                agent.store_reward(reward)

            else:

                move = opponent_agent.decide_next_move(engine)

                # 상대 메모리는 학습 불필요하므로 즉시 비움
                opponent_agent.memory.clear()

                if move is None:
                    abnormal_end = True
                    break

                success = engine.make_move(*move)

                if not success:
                    agent.store_reward(1.0)
                    abnormal_end = True
                    break

        # 결과 집계
        if not abnormal_end:

            if engine.winner == PLAYER_1:
                total_wins    += 1
                interval_wins += 1

            elif engine.winner == 0:
                total_draws    += 1
                interval_draws += 1

            else:
                total_losses    += 1
                interval_losses += 1

        else:
            total_losses    += 1
            interval_losses += 1

        # PPO 업데이트
        agent.update()

        # 주기적 로그 출력
        if (episode + 1) % LOG_INTERVAL == 0:

            interval_total = (
                interval_wins
                + interval_losses
                + interval_draws
            )

            win_rate = (
                (interval_wins / interval_total) * 100
                if interval_total > 0 else 0.0
            )

            print(
                'Episode {}/{} | '
                '승 {} | 패 {} | 무 {} | '
                '구간 승률 {:.1f}% | '
                '전체 승률 {:.1f}%'
                .format(
                    episode + 1,
                    NUM_EPISODES,
                    interval_wins,
                    interval_losses,
                    interval_draws,
                    win_rate,
                    (total_wins / (episode + 1)) * 100,
                )
            )

            # 최고 승률 갱신 시 저장
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                print(
                    '최고 승률 갱신 ({:.1f}%) → 모델 저장'
                    .format(best_win_rate)
                )
                agent.save()

            # 구간 승률 55% 이상 시 상대 모델 갱신
            if win_rate > 55.0:
                opponent_agent.policy.load_state_dict(
                    copy.deepcopy(agent.policy.state_dict())
                )
                opponent_agent.policy_old.load_state_dict(
                    opponent_agent.policy.state_dict()
                )
                print(
                    '상대 모델 갱신 (구간 승률 {:.1f}%)'
                    .format(win_rate)
                )

            # 구간 카운터 리셋
            interval_wins   = 0
            interval_losses = 0
            interval_draws  = 0

    print('-' * 50)
    print('학습 완료')
    print(
        '최종 승률: {:.1f}%'
        .format((total_wins / NUM_EPISODES) * 100)
    )
    agent.save()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    train()