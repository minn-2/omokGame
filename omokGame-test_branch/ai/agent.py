import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.Rules import Rules
from ai.model import PPOModel


# 경험 버퍼
class Memory:
    def __init__(self):
        self.states   : list = []
        self.actions  : list = []
        self.logprobs : list = []
        self.values   : list = []
        self.rewards  : list = []
        self.dones    : list = []

    def clear(self):
        for lst in (self.states, self.actions, self.logprobs,
                    self.values, self.rewards, self.dones):
            lst.clear()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    # ── 하이퍼파라미터
    LR           = 3e-4
    GAMMA        = 0.99
    GAE_LAMBDA   = 0.95
    CLIP_EPS     = 0.2
    ENTROPY_C    = 0.005
    VALUE_C      = 0.5
    EPOCHS       = 4
    BATCH_SIZE   = 64
    TRAIN_EVERY  = 1
    REWARD_SCALE = 0.05

    # 후보 칸 탐색 반경
    NEIGHBOR_RADIUS = 2

    def __init__(self, board_size: int = 15, player: int = 2):
        self.board_size   = board_size
        self.player       = player
        self._eval_mode   = False
        self.device       = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net     = PPOModel(board_size).to(self.device)
        self.old_net = PPOModel(board_size).to(self.device)
        self.old_net.load_state_dict(self.net.state_dict())
        self.old_net.eval()

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.LR)

        self.memory    = Memory()
        self.last_loss : float | None = None
        self._ep_count = 0

    # ── 행동 선택
    def decide_next_move(self, engine):
        state   = engine.get_state()
        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value = self.old_net(state_t)

        probs = probs.squeeze(0)
        value = value.item()

        board_np  = engine.board.board
        cur       = engine.current_player

        # 후보 마스크 + 거리 가중치
        mask_flat, dist_weight = self._build_mask(board_np, cur)

        if not mask_flat.any():
            return None

        # 정책 확률 × 거리 가중치 → 가까운 곳에 높은 확률 부여
        probs = probs * mask_flat.float() * dist_weight
        s     = probs.sum()
        probs = probs / s if s > 0 \
            else mask_flat.float() / mask_flat.float().sum()

        dist   = Categorical(probs)
        action = dist.sample()

        if not self._eval_mode:
            self.memory.states.append(state)
            self.memory.actions.append(action.item())
            self.memory.logprobs.append(dist.log_prob(action).item())
            self.memory.values.append(value)

        row, col = divmod(action.item(), self.board_size)
        return (row, col)

    def _build_mask(self, board_np: np.ndarray,
                    player: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        착수 후보 마스크 + 거리 가중치 반환.

        - 흑·백 모두 기존 돌 주변 NEIGHBOR_RADIUS 칸 이내만 후보
        - 가중치: 가까울수록 높음 (체비쇼프 거리 기준 1/dist)
        - 흑돌은 추가로 금수 필터 적용
        - 보드가 비어 있으면 중앙만 후보

        Returns
        -------
        mask        : (n*n,) bool tensor
        dist_weight : (n*n,) float tensor (정규화됨)
        """
        n        = self.board_size
        mask     = (board_np == 0).flatten()
        weight   = np.zeros(n * n, dtype=np.float32)
        occupied = np.argwhere(board_np != 0)

        if len(occupied) == 0:
            # 보드가 비어 있으면 중앙만 후보
            center      = n // 2
            idx         = center * n + center
            mask        = np.zeros(n * n, dtype=bool)
            mask[idx]   = True
            weight[idx] = 1.0
        else:
            candidate_mask = np.zeros(n * n, dtype=bool)
            r_rad          = self.NEIGHBOR_RADIUS

            for or_, oc in occupied:
                for dr in range(-r_rad, r_rad + 1):
                    for dc in range(-r_rad, r_rad + 1):
                        nr, nc = int(or_) + dr, int(oc) + dc
                        if (0 <= nr < n and 0 <= nc < n
                                and board_np[nr, nc] == 0):
                            idx            = nr * n + nc
                            candidate_mask[idx] = True
                            # 체비쇼프 거리: 가까울수록 가중치 높음
                            d = max(abs(dr), abs(dc))   # d >= 1 보장
                            w = 1.0 / d
                            if w > weight[idx]:
                                weight[idx] = w

            mask = mask & candidate_mask

            # 흑돌 금수 필터
            if player == 1:
                for idx in np.where(mask)[0]:
                    r, c = divmod(int(idx), n)
                    if Rules.is_forbidden(board_np, r, c, 1):
                        mask[idx]   = False
                        weight[idx] = 0.0

        # 가중치 정규화 (마스크 밖은 0)
        weight = weight * mask.astype(np.float32)
        w_sum  = weight.sum()
        if w_sum > 0:
            weight /= w_sum

        mask_t   = torch.tensor(mask,   dtype=torch.bool,   device=self.device)
        weight_t = torch.tensor(weight, dtype=torch.float32, device=self.device)
        return mask_t, weight_t

    # ── 보상 저장 + 학습 트리거
    def store_reward(self, reward: float, done: bool = False):
        if self._eval_mode:
            return
        scaled_reward = reward * self.REWARD_SCALE
        self.memory.rewards.append(scaled_reward)
        self.memory.dones.append(done)

        if done:
            self._ep_count += 1
            if self._ep_count % self.TRAIN_EVERY == 0:
                self.last_loss = self._train()
                self.memory.clear()

    # ── PPO 학습
    def _train(self) -> float | None:
        T = min(len(self.memory.states), len(self.memory.rewards))
        if T < self.BATCH_SIZE:
            return self.last_loss

        states   = torch.tensor(
            np.array(self.memory.states[:T]),
            dtype=torch.float32, device=self.device)
        actions  = torch.tensor(
            self.memory.actions[:T],
            dtype=torch.long, device=self.device)
        old_lps  = torch.tensor(
            self.memory.logprobs[:T],
            dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(
            self.memory.values[:T],
            dtype=torch.float32, device=self.device)
        rewards  = self.memory.rewards[:T]
        dones    = self.memory.dones[:T]

        advantages = self._gae(rewards, dones, old_vals)
        returns    = (advantages + old_vals).detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8)

        total_loss, count = 0.0, 0
        for _ in range(self.EPOCHS):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, self.BATCH_SIZE):
                mb = idx[start: start + self.BATCH_SIZE]
                if len(mb) < 2:
                    continue
                loss = self._ppo_loss(
                    states[mb], actions[mb],
                    old_lps[mb], advantages[mb], returns[mb])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item()
                count      += 1

        self.old_net.load_state_dict(self.net.state_dict())
        return total_loss / max(count, 1)

    def _gae(self, rewards, dones, values) -> torch.Tensor:
        T    = len(rewards)
        advs = torch.zeros(T, device=self.device)
        gae  = 0.0
        for t in reversed(range(T)):
            nv    = values[t+1].item() if t+1 < T else 0.0
            delta = (rewards[t]
                     + self.GAMMA * nv * (1 - int(dones[t]))
                     - values[t].item())
            gae   = (delta
                     + self.GAMMA * self.GAE_LAMBDA
                     * (1 - int(dones[t])) * gae)
            advs[t] = gae
        return advs

    def _ppo_loss(self, states, actions, old_log_probs,
                  advantages, returns) -> torch.Tensor:
        probs, values = self.net(states)
        values        = values.squeeze(-1)
        dist          = Categorical(probs)
        log_probs     = dist.log_prob(actions)
        entropy       = dist.entropy().mean()

        ratio       = (log_probs - old_log_probs).exp()
        clip_ratio  = ratio.clamp(1-self.CLIP_EPS, 1+self.CLIP_EPS)
        policy_loss = -torch.min(
            ratio * advantages, clip_ratio * advantages).mean()
        value_loss  = F.mse_loss(values, returns)

        return (policy_loss
                + self.VALUE_C   * value_loss
                - self.ENTROPY_C * entropy)

    # ── 가중치 관리
    def save(self, path: str = 'ppo_p2.pt'):
        torch.save({
            'net'      : self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f'[PPOAgent] 저장 → {path}')

    def load(self, path: str = 'ppo_p2.pt'):
        if not os.path.exists(path):
            print(f'[PPOAgent] {path} 없음 — 랜덤 가중치로 시작')
            return
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
        self.old_net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'[PPOAgent] 불러오기 ← {path}')

    def clone_weights(self) -> dict:
        return {k: v.cpu().clone()
                for k, v in self.net.state_dict().items()}

    def make_champion_agent(self, weights: dict) -> 'PPOAgent':
        champ = PPOAgent(self.board_size, player=2)
        sd    = {k: v.to(self.device) for k, v in weights.items()}
        champ.net.load_state_dict(sd)
        champ.old_net.load_state_dict(sd)
        champ.old_net.eval()
        return champ