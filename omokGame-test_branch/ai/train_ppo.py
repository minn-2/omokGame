import numpy as np
import torch
import copy

from ai.engine import Engine
from ai.agent import PPOAgent


BOARD_SIZE = 15
NUM_EPISODES = 200000
LOG_INTERVAL = 100
OPPONENT_UPDATE_INTERVAL = 500

PLAYER_1 = 1
PLAYER_2 = 2


def train():

    engine = Engine(BOARD_SIZE)

    agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_1,
        in_channels=1          
    )

    opponent_agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER_2,
        in_channels=1         
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

                # calculate_reward가 종료 보상도 포함하므로 그대로 사용
                reward = agent.calculate_reward(engine)
                agent.store_reward(reward)

            else:

                # 상대는 학습하지 않으므로 메모리 저장 없이 행동만 선택
                move = opponent_agent.decide_next_move(engine)

                # 상대 메모리는 매 턴 즉시 비움 (메모리 낭비 방지)
                opponent_agent.memory.clear()

                if move is None:
                    abnormal_end = True
                    break

                success = engine.make_move(*move)

                if not success:
                    agent.store_reward(1.0)
                    abnormal_end = True
                    break

        # 정상 종료 시에만 결과 집계
        if not abnormal_end:

            # alculate_reward가 이미 종료 보상을 저장했으므로
            # 중복 저장 없이 집계만 수행
            if engine.winner == PLAYER_1:
                total_wins   += 1
                interval_wins += 1

            elif engine.winner == 0:
                total_draws   += 1
                interval_draws += 1

            else:
                total_losses   += 1
                interval_losses += 1

        else:
            # 비정상 종료는 패배로 집계
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

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                print(
                    '최고 승률 갱신 ({:.1f}%) → 모델 저장'
                    .format(best_win_rate)
                )
                agent.save()

            interval_wins   = 0
            interval_losses = 0
            interval_draws  = 0

        # 주기적 상대 모델 갱신
        if (episode + 1) % OPPONENT_UPDATE_INTERVAL == 0:

            opponent_agent.policy.load_state_dict(
                copy.deepcopy(agent.policy.state_dict())
            )
            opponent_agent.policy_old.load_state_dict(
                opponent_agent.policy.state_dict()
            )

            print(
                'Self-Play 상대 모델 갱신 완료 (Episode {})'
                .format(episode + 1)
            )

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