import numpy as np
from ai.engine  import Engine
from ai.agent   import PPOAgent

# 학습 하이퍼파라미터
BOARD_SIZE   = 15
NUM_EPISODES = 10000
PLAYER       = 2      # AI = 백돌

def train():
    engine = Engine(BOARD_SIZE)
    agent  = PPOAgent(BOARD_SIZE, player=PLAYER)

    wins   = 0
    losses = 0

    print('PPO 학습 시작...')
    print('보드 크기: {}x{}'.format(BOARD_SIZE, BOARD_SIZE))
    print('총 에피소드: {}'.format(NUM_EPISODES))
    print('-' * 40)

    for episode in range(NUM_EPISODES):
        engine.reset()

        while not engine.is_over:
            if engine.current_player == PLAYER:
                # PPO 착수
                move = agent.decide_next_move(engine)
                if move is None:
                    break
                engine.make_move(*move)
                reward = agent.calculate_reward(engine)
                agent.store_reward(reward)
            else:
                # 상대방: 랜덤 착수
                valid = engine.get_valid_moves()
                if len(valid) == 0:
                    break
                idx = np.random.randint(len(valid))
                engine.make_move(*valid[idx])

        # 결과 기록
        if engine.winner == PLAYER:
            wins += 1
        elif engine.winner != 0:
            losses += 1

        # PPO 업데이트
        agent.update()

        # 100번마다 출력 및 저장
        if (episode + 1) % 100 == 0:
            total = episode + 1
            print('Episode {}/{} | 승: {} | 패: {} | 승률: {:.1f}%'
                  .format(total, NUM_EPISODES,
                          wins, losses,
                          wins / total * 100))
            agent.save()

    print('-' * 40)
    print('학습 완료!')
    agent.save()

if __name__ == '__main__':
    train()