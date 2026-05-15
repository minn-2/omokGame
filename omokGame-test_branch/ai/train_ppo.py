import numpy as np
import torch

from ai.engine import Engine
from ai.agent import PPOAgent


# 설정
BOARD_SIZE = 15
NUM_EPISODES = 1000
PLAYER = 2


def train():
    # 엔진 / PPO 에이전트 생성
    engine = Engine(BOARD_SIZE)

    agent = PPOAgent(
        BOARD_SIZE,
        player=PLAYER
    )

    # 통계 변수
    wins = 0
    losses = 0
    draws = 0

    best_win_rate = 0.0

    print('PPO 학습 시작')

    print(
        'Device:',
        agent.device
    )

    print('-' * 50)

    # 학습 루프
    for episode in range(NUM_EPISODES):

        # 게임 초기화
        engine.reset()

        while not engine.is_over:

            # PPO AI 차례
            if engine.current_player == PLAYER:

                move = agent.decide_next_move(
                    engine
                )

                # 둘 곳 없으면 종료
                if move is None:
                    break

                # 착수
                success = engine.make_move(
                    *move
                )

                # 실패 시 종료
                if not success:
                    break

                # 보상 계산
                reward = (
                    agent.calculate_reward(
                        engine
                    )
                )

                # 보상 저장
                agent.store_reward(reward)

            # 랜덤 상대 차례
            else:

                valid_moves = (
                    engine.get_valid_moves()
                )

                # 가능한 수 없으면 종료
                if len(valid_moves) == 0:
                    break

                # 랜덤 위치 선택
                idx = np.random.randint(
                    len(valid_moves)
                )

                move = valid_moves[idx]

                engine.make_move(
                    *move
                )

        # 결과 기록
        if engine.winner == PLAYER:

            wins += 1

        elif engine.winner == 0:

            draws += 1

        else:

            losses += 1

        # PPO 업데이트
        agent.update()

        # 출력 및 저장
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

                .format(total, NUM_EPISODES, wins, losses, draws, win_rate)
            )

            # 최고 승률 갱신 시 저장
            if win_rate > best_win_rate:

                best_win_rate = win_rate

                print(
                    '최고 승률 갱신 → 모델 저장'
                )

                agent.save()

    # 학습 종료
    print('-' * 50)

    print('학습 완료')

    print(
        '최종 승률: {:.1f}%'
        .format(
            (wins / NUM_EPISODES) * 100
        )
    )

    # 최종 저장
    agent.save()


# 실행
if __name__ == '__main__':

    # 랜덤 시드 고정
    np.random.seed(42)
    
    torch.manual_seed(42)

    train()
