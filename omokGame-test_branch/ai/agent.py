import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.distributions import Categorical
from ai.model import PPOModel


class Memory:
    """경험 저장 버퍼"""

    def __init__(self):

        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear(self):

        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()


class PPOAgent:

    def __init__(
            self,
            board_size=15,
            player=2,
            lr=1e-4,
            gamma=0.99,
            eps_clip=0.1,
            in_channels=3):         
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        self.board_size = board_size
        self.player = player
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.in_channels = in_channels

        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self.weights_path = os.path.join(
            BASE_DIR,
            'models',
            'ppo_p{}.pt'.format(player)
        )

        # in_channels 전달
        self.policy = PPOModel(
            board_size,
            in_channels=in_channels
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

        # in_channels 전달
        self.policy_old = PPOModel(
            board_size,
            in_channels=in_channels
        ).to(self.device)

        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        if os.path.isfile(self.weights_path):

            print(
                '모델 불러오기:',
                self.weights_path
            )

            self.policy.load_state_dict(
                torch.load(
                    self.weights_path,
                    map_location=self.device,
                    weights_only=True       # 보안 경고 제거
                )
            )

            self.policy_old.load_state_dict(
                self.policy.state_dict()
            )

        else:

            print('새 모델 생성')

    # 행동 선택
    def select_action(
            self,
            state,
            valid_moves):

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        # (C,H,W) -> (1,C,H,W) 배치 차원 추가
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            probs, _ = self.policy_old(state)

        mask = torch.zeros(
            self.board_size * self.board_size
        ).to(self.device)

        for move in valid_moves:

            idx = int(
                move[0] * self.board_size
                + move[1]
            )

            mask[idx] = 1

        masked_probs = probs.squeeze(0) * mask  # squeeze(0)으로 배치만 제거

        sum_probs = masked_probs.sum()

        if sum_probs > 0:
            masked_probs = masked_probs / sum_probs
        else:
            masked_probs = mask / (mask.sum() + 1e-8)

        dist = Categorical(masked_probs)

        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action)
        )

    # 다음 수 결정
    def decide_next_move(
            self,
            engine):

        state = engine.get_state()

        valid_moves = engine.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        action, log_prob = self.select_action(
            state,
            valid_moves
        )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        # (C,H,W) 형태로 메모리에 저장
        # get_state()가 (C,H,W)를 반환하면 그대로,
        # (1,C,H,W)를 반환하면 배치 차원 제거
        if state_tensor.dim() == 4:
            state_tensor = state_tensor.squeeze(0)

        self.memory.states.append(state_tensor)

        self.memory.actions.append(
            torch.tensor(
                action,
                dtype=torch.long
            ).to(self.device)
        )

        self.memory.logprobs.append(
            log_prob.detach()
        )

        row = action // self.board_size
        col = action % self.board_size

        return row, col

    # 보상 저장
    def store_reward(self, reward):

        self.memory.rewards.append(reward)

    # 보상 계산
    # agent.py calculate_reward() 수정
    def calculate_reward(self, engine):

        if engine.is_over:
            if engine.winner == self.player:
                return 100.0   # 승리 보상 상향
            elif engine.winner == 0:
                return -10.0   # 무승부는 패배에 가깝게
            else:
                return -100.0  # 패배 패널티 상향

        reward = 0.0
        opponent = 3 - self.player

        num_my_5 = engine.check_patterns(self.player, 5)
        num_my_4 = engine.check_patterns(self.player, 4)
        num_my_3 = engine.check_patterns(self.player, 3)

        num_op_5 = engine.check_patterns(opponent, 5)
        num_op_4 = engine.check_patterns(opponent, 4)
        num_op_3 = engine.check_patterns(opponent, 3)

        # 공격 보상
        reward += num_my_5 * 50.0  # 5목 직전
        reward += num_my_4 * 5.0
        reward += num_my_3 * 1.5

        # 수비 패널티 (공격보다 수비를 더 중요하게)
        reward -= num_op_5 * 60.0  # 상대 5목 직전은 반드시 막아야
        reward -= num_op_4 * 8.0
        reward -= num_op_3 * 2.0

        return max(min(reward, 50.0), -50.0)

    # PPO 업데이트
    def update(self):

        if len(self.memory.rewards) == 0:
            return

        # 할인 누적 보상 계산
        discounted_rewards = []
        discounted_reward = 0

        for reward in reversed(self.memory.rewards):

            discounted_reward = (
                reward + self.gamma * discounted_reward
            )

            discounted_rewards.insert(0, discounted_reward)

        # 데이터 길이 동기화
        min_size = min(
            len(self.memory.states),
            len(self.memory.actions),
            len(self.memory.logprobs),
            len(discounted_rewards)
        )

        # 경험이 없으면 업데이트 스킵
        if min_size == 0:
            self.memory.clear()
            return

        self.memory.states   = self.memory.states[:min_size]
        self.memory.actions  = self.memory.actions[:min_size]
        self.memory.logprobs = self.memory.logprobs[:min_size]
        discounted_rewards   = discounted_rewards[:min_size]

        # Tensor 변환
        # stack 결과: [Batch, C, H, W]
        states = torch.stack(
            self.memory.states
        ).to(self.device).detach()

        actions = torch.stack(
            self.memory.actions
        ).to(self.device).detach()

        logprobs = torch.stack(
            self.memory.logprobs
        ).to(self.device).detach()

        rewards = torch.tensor(
            discounted_rewards,
            dtype=torch.float32
        ).to(self.device).detach()

        # 보상 정규화
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # PPO 학습
        for _ in range(4):

            probs, state_values = self.policy(states)

            state_values = state_values.squeeze(-1)  # [Batch,1] -> [Batch]

            dist = Categorical(probs)

            new_logprobs = dist.log_prob(actions)

            entropy = dist.entropy()

            ratios = torch.exp(new_logprobs - logprobs)

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages

            surr2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * entropy
            )

            self.optimizer.zero_grad()

            loss.mean().backward()

            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                0.5
            )

            self.optimizer.step()

        # 이전 정책 갱신
        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

        self.memory.clear()

    # 모델 저장
    def save(self):

        os.makedirs(
            os.path.dirname(self.weights_path),
            exist_ok=True
        )

        torch.save(
            self.policy.state_dict(),
            self.weights_path
        )

        print('모델 저장:', self.weights_path)