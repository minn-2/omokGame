import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from ai.model import PPOModel


class Memory:
    """경험 재생 버퍼"""
    def __init__(self):
        self.states   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()


class PPOAgent:
    def __init__(self, board_size=15, player=2,
                 lr=3e-4, gamma=0.99, eps_clip=0.2):
        """
        PPO 에이전트
        player:   1=흑돌, 2=백돌
        lr:       학습률
        gamma:    할인율
        eps_clip: PPO 클리핑 범위
        """
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.board_size   = board_size
        self.player       = player
        self.gamma        = gamma
        self.eps_clip     = eps_clip
        self.weights_path = './models/ppo_p{}.pt'.format(player)

        # Policy / Policy_old
        self.policy     = PPOModel(board_size).to(self.device)
        self.optimizer  = optim.Adam(
            self.policy.parameters(), lr=lr)
        self.policy_old = PPOModel(board_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.memory  = Memory()

        # 저장된 모델 불러오기
        if os.path.isfile(self.weights_path):
            print('모델 불러오는 중: {}'.format(self.weights_path))
            self.policy.load_state_dict(
                torch.load(self.weights_path,
                           map_location=self.device))
            self.policy_old.load_state_dict(
                self.policy.state_dict())
        else:
            print('새 모델로 시작합니다')

    def select_action(self, state, valid_moves):
        """
        착수 위치 결정
        get_valid_moves()로 이미 돌이 있는 곳 마스킹
        """
        state = torch.FloatTensor(state)\
            .unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, _ = self.policy_old(state)

        # 유효한 위치만 마스킹
        mask = torch.zeros(
            self.board_size * self.board_size).to(self.device)
        for m in valid_moves:
            mask[m[0] * self.board_size + m[1]] = 1

        masked_probs = probs.squeeze() * mask

        # 확률 정규화
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = mask / mask.sum()

        # 확률적으로 행동 선택
        dist   = Categorical(masked_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def decide_next_move(self, engine):
        """GUI / 학습에서 호출하는 착수 결정 함수"""
        state       = engine.get_state()
        valid_moves = engine.get_valid_moves()

        if len(valid_moves) == 0:
            return None

        action, log_prob = self.select_action(state, valid_moves)

        # 경험 저장
        self.memory.states.append(
            torch.FloatTensor(state).to(self.device))
        self.memory.actions.append(
            torch.tensor(action).to(self.device))
        self.memory.logprobs.append(log_prob)

        row = action // self.board_size
        col = action % self.board_size
        return row, col

    def store_reward(self, reward):
        self.memory.rewards.append(reward)

    def calculate_reward(self, engine):
        """
        보상 계산
        승리:   +1.0
        패배:   -1.0
        무승부: +0.3
        4목:    +0.5
        3목:    +0.3
        """
        if engine.is_over:
            if engine.winner == self.player:
                return 1.0
            elif engine.winner == 0:
                return 0.3
            else:
                return -1.0
        if engine.check_patterns(self.player, 4):
            return 0.5
        if engine.check_patterns(self.player, 3):
            return 0.3
        return 0.0

    def update(self):
        """PPO 핵심 업데이트"""
        if len(self.memory.rewards) == 0:
            return

        states   = torch.stack(
            self.memory.states).to(self.device).detach()
        actions  = torch.stack(
            self.memory.actions).to(self.device).detach()
        logprobs = torch.stack(
            self.memory.logprobs).to(self.device).detach()
        rewards  = torch.tensor(
            self.memory.rewards,
            dtype=torch.float32).to(self.device).detach()

        # 보상 정규화
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / \
                      (rewards.std() + 1e-5)

        # 10 에포크 반복 학습
        for _ in range(10):
            probs, state_values = self.policy(states)
            dist         = Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            entropy      = dist.entropy()

            # Ratio: 이전 정책 대비 현재 정책 변화율
            ratios     = torch.exp(new_logprobs - logprobs)

            # Advantage: 실제 보상 - 예측 가치
            advantages = rewards - state_values.detach().squeeze()

            # PPO Clipped Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(
                ratios,
                1 - self.eps_clip,
                1 + self.eps_clip) * advantages

            # 최종 손실
            # 정책손실 + 가치손실 - 엔트로피 보너스
            loss = (-torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(
                        state_values.squeeze(), rewards)
                    - 0.01 * entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # policy_old 업데이트
        self.policy_old.load_state_dict(
            self.policy.state_dict())
        self.memory.clear()

    def save(self):
        os.makedirs('./models', exist_ok=True)
        torch.save(self.policy.state_dict(), self.weights_path)
        print('모델 저장: {}'.format(self.weights_path))