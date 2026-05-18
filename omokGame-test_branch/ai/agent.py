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
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2):

        # GPU 사용 가능 시 CUDA 사용
        self.device = torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        self.board_size = board_size
        self.player = player
        self.gamma = gamma
        self.eps_clip = eps_clip

        # 프로젝트 루트 경로
        BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        # 모델 저장 경로
        self.weights_path = os.path.join(
            BASE_DIR,
            'models',
            'ppo_p{}.pt'.format(player)
        )

        # PPO 현재 정책
        self.policy = PPOModel(
            board_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

        # PPO 이전 정책
        self.policy_old = PPOModel(
            board_size
        ).to(self.device)

        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

        # Loss 함수
        self.MseLoss = nn.MSELoss()

        # 경험 메모리
        self.memory = Memory()

        # 저장된 모델 불러오기
        if os.path.isfile(self.weights_path):

            print(
                '모델 불러오기:',
                self.weights_path
            )

            self.policy.load_state_dict(
                torch.load(
                    self.weights_path,
                    map_location=self.device
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

        # Tensor 변환
        state = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        # 배치 차원 추가
        if state.dim() == 3:

            state = state.unsqueeze(0)

        # 이전 정책 추론
        with torch.no_grad():

            probs, _ = self.policy_old(state)

        # 가능한 위치만 허용
        mask = torch.zeros(
            self.board_size * self.board_size
        ).to(self.device)

        for move in valid_moves:

            idx = int(
                move[0] * self.board_size
                + move[1]
            )

            mask[idx] = 1

        # 불가능한 위치 제거
        masked_probs = probs.squeeze() * mask

        # 확률 정규화
        sum_probs = masked_probs.sum()

        if sum_probs > 0:

            masked_probs = (
                masked_probs / sum_probs
            )

        else:

            masked_probs = (
                mask / (mask.sum() + 1e-8)
            )

        # 확률 분포 생성
        dist = Categorical(masked_probs)

        # 행동 선택
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

        # 가능한 수 없으면 종료
        if len(valid_moves) == 0:

            return None

        # 행동 선택
        action, log_prob = self.select_action(
            state,
            valid_moves
        )

        # 상태 저장
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        # [1,1,15,15] -> [1,15,15]
        if (
            state_tensor.dim() == 4
            and state_tensor.size(0) == 1
        ):

            state_tensor = state_tensor.squeeze(0)

        self.memory.states.append(
            state_tensor
        )

        # 행동 저장
        self.memory.actions.append(

            torch.tensor(
                action,
                dtype=torch.long
            ).to(self.device)
        )

        # 로그 확률 저장
        self.memory.logprobs.append(
            log_prob.detach()
        )

        # 1차원 행동 -> 2차원 좌표
        row = action // self.board_size
        col = action % self.board_size

        return row, col

    # 보상 저장
    def store_reward(self, reward):

        self.memory.rewards.append(reward)

    # 보상 계산
    def calculate_reward(self, engine):

        # 게임 종료 보상
        if engine.is_over:

            if engine.winner == self.player:

                return 50.0

            elif engine.winner == 0:

                return 5.0

            else:

                return -50.0

        reward = 0.0

        opponent = 3 - self.player

        # 공격 패턴
        num_my_4 = engine.check_patterns(
            self.player,
            4
        )

        num_my_3 = engine.check_patterns(
            self.player,
            3
        )

        # 상대 패턴
        num_op_4 = engine.check_patterns(
            opponent,
            4
        )

        num_op_3 = engine.check_patterns(
            opponent,
            3
        )

        # 공격 보상
        reward += num_my_4 * 2.0
        reward += num_my_3 * 0.8

        # 수비 패널티
        reward -= num_op_4 * 4.0
        reward -= num_op_3 * 1.5

        # Reward clipping
        return max(
            min(reward, 20.0),
            -20.0
        )

    # PPO 업데이트
    def update(self):

        # 경험 없으면 종료
        if len(self.memory.rewards) == 0:

            return

        # 할인 누적 보상 계산
        discounted_rewards = []

        discounted_reward = 0

        for reward in reversed(
                self.memory.rewards):

            discounted_reward = (
                reward
                + self.gamma
                * discounted_reward
            )

            discounted_rewards.insert(
                0,
                discounted_reward
            )

        # 데이터 길이 동기화
        min_size = min(
            len(self.memory.states),
            len(self.memory.actions),
            len(self.memory.logprobs),
            len(discounted_rewards)
        )

        self.memory.states = (
            self.memory.states[:min_size]
        )

        self.memory.actions = (
            self.memory.actions[:min_size]
        )

        self.memory.logprobs = (
            self.memory.logprobs[:min_size]
        )

        discounted_rewards = (
            discounted_rewards[:min_size]
        )

        # Tensor 변환
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

            rewards = (
                rewards - rewards.mean()
            ) / (
                rewards.std() + 1e-5
            )

        # PPO 학습
        for _ in range(4):

            probs, state_values = (
                self.policy(states)
            )

            # [Batch,1] -> [Batch]
            state_values = (
                state_values.squeeze(-1)
            )

            dist = Categorical(probs)

            new_logprobs = (
                dist.log_prob(actions)
            )

            entropy = dist.entropy()

            # PPO Ratio
            ratios = torch.exp(
                new_logprobs - logprobs
            )

            # Advantage
            advantages = (
                rewards
                - state_values.detach()
            )

            # PPO Objective
            surr1 = ratios * advantages

            surr2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip
            ) * advantages

            # Loss
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(
                    state_values,
                    rewards
                )
                - 0.01 * entropy
            )

            # 역전파
            self.optimizer.zero_grad()

            loss.mean().backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                0.5
            )

            self.optimizer.step()

        # 이전 정책 갱신
        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

        # 메모리 초기화
        self.memory.clear()

    # 모델 저장
    def save(self):

        # models 폴더 생성
        os.makedirs(
            os.path.dirname(
                self.weights_path
            ),
            exist_ok=True
        )

        # 모델 저장
        torch.save(
            self.policy.state_dict(),
            self.weights_path
        )

        print(
            '모델 저장:',
            self.weights_path
        )
