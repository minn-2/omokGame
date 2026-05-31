import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.Rules import Rules
from ai.model  import PPOModel

MCTS_SIMS_THREAT      = 25    # 위협 감지 시 시뮬레이션 수
MCTS_SIMS_LATE        = 30    # 후반 시뮬레이션 수
LATE_GAME_THRESHOLD   = 30    # 이 수 이후를 "후반"으로 판단
C_PUCT                = 1.5


# ──────────────────────────────────────────
# 경험 버퍼
# ──────────────────────────────────────────

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


# ──────────────────────────────────────────
# 인라인 얕은 MCTS (외부 파일 없이 agent 내부에 완전 내장)
# ──────────────────────────────────────────

class _Node:
    __slots__ = ('parent','action','children',
                 'n','w','p','expanded')
    def __init__(self, parent, action, prior):
        self.parent   = parent
        self.action   = action
        self.children : dict = {}
        self.n = 0; self.w = 0.0; self.p = prior
        self.expanded = False

    @property
    def q(self):
        return self.w / self.n if self.n > 0 else 0.0

    def ucb(self, c):
        u = c * self.p * math.sqrt(self.parent.n) / (1 + self.n)
        return self.q + u

    def best_child(self, c):
        return max(self.children.values(), key=lambda x: x.ucb(c))


def _check_win(board, action, player, n):
    r, c = divmod(action, n)
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt = 1
        for s in (1,-1):
            nr, nc = r+dr*s, c+dc*s
            while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                cnt+=1; nr+=dr*s; nc+=dc*s
        if cnt >= 5:
            return True
    return False


def _valid_mask(board, player, n):
    """빈 칸 중 기존 돌 주변 2칸 이내 + 흑돌 금수 제거."""
    flat = (board == 0).flatten()
    occupied = np.argwhere(board != 0)
    if len(occupied) == 0:
        cand = np.zeros(n*n, dtype=bool)
        c = n//2; cand[c*n+c] = True
    else:
        cand = np.zeros(n*n, dtype=bool)
        for or_, oc in occupied:
            r0=max(0,int(or_)-2); r1=min(n,int(or_)+3)
            c0=max(0,int(oc)-2);  c1=min(n,int(oc)+3)
            for nr in range(r0,r1):
                for nc in range(c0,c1):
                    if board[nr,nc]==0:
                        cand[nr*n+nc]=True
    mask = flat & cand
    if player == 1:
        for idx in np.where(mask)[0]:
            r,c = divmod(int(idx),n)
            if Rules.is_forbidden(board,r,c,1):
                mask[idx] = False
    return mask


def _shallow_mcts(net, board_np, player, n_sims, n, device):
    """
    얕은 MCTS. 최선 action(flat index) 반환.
    net: old_net (eval 모드)
    """
    root = _Node(None, -1, 1.0)
    root.n = 1  # sqrt(0) 방지

    def _expand(node, b, p):
        mask = _valid_mask(b, p, n)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            node.expanded = True; return

        # 정책 네트워크 prior
        ch0 = (b==p).astype(np.float32)
        ch1 = (b==(3-p)).astype(np.float32)
        ch2 = np.full((n,n), 1.0 if p==1 else 0.0, dtype=np.float32)
        st  = torch.tensor(np.stack([ch0,ch1,ch2]),
                           dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = net(st)
        pr = probs.squeeze(0).cpu().numpy()
        pr[~mask] = 0.0
        s = pr.sum()
        if s > 1e-8: pr /= s
        else: pr[idxs] = 1.0/len(idxs)

        for idx in idxs:
            node.children[int(idx)] = _Node(node, int(idx), float(pr[idx]))
        node.expanded = True

    def _evaluate(b, p):
        ch0=(b==p).astype(np.float32)
        ch1=(b==(3-p)).astype(np.float32)
        ch2=np.full((n,n),1.0 if p==1 else 0.0,dtype=np.float32)
        st=torch.tensor(np.stack([ch0,ch1,ch2]),
                        dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v = net(st)
        return float(v.item())

    _expand(root, board_np, player)

    for _ in range(n_sims):
        node = root
        b    = board_np.copy()
        p    = player

        # Selection
        while node.expanded and node.children:
            node = node.best_child(C_PUCT)
            r,c  = divmod(node.action, n)
            b[r,c] = p; p = 3-p

        # Terminal check
        if node.action >= 0 and _check_win(b, node.action, 3-p, n):
            v = -1.0
        elif not (b==0).any():
            v = 0.0
        else:
            if not node.expanded:
                _expand(node, b, p)
                if node.children:
                    node = node.best_child(C_PUCT)
                    r,c  = divmod(node.action, n)
                    b[r,c] = p; p = 3-p
            v = _evaluate(b, p)

        # Backprop
        cur = node
        while cur is not None:
            cur.n += 1; cur.w += v; v = -v; cur = cur.parent

    if not root.children:
        return None
    # 방문 횟수 최대 노드 선택
    best = max(root.children.values(), key=lambda x: x.n)
    return best.action


# ──────────────────────────────────────────
# 위협 감지 유틸
# ──────────────────────────────────────────

def _max_len_at(board, r, c, player, n):
    best = 1
    for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt=1
        for s in (1,-1):
            nr,nc=r+dr*s,c+dc*s
            while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                cnt+=1; nr+=dr*s; nc+=dc*s
        best=max(best,cnt)
    return best


def _threat_level(board, last_move, player, n):
    if last_move is None:
        return 0
    r, c = last_move
    return _max_len_at(board, r, c, player, n)


# ──────────────────────────────────────────
# PPOAgent
# ──────────────────────────────────────────

class PPOAgent:
    LR           = 3e-4
    GAMMA        = 0.99
    GAE_LAMBDA   = 0.95
    CLIP_EPS     = 0.2
    ENTROPY_C    = 0.01
    VALUE_C      = 0.5
    EPOCHS       = 4
    BATCH_SIZE   = 16     # 128 → 16: 짧은 게임에서도 학습 보장
    TRAIN_EVERY  = 1
    REWARD_SCALE = 0.05

    def __init__(self, board_size: int = 15, player: int = 2):
        self.board_size = board_size
        self.player     = player
        self._eval_mode = False
        self.device     = torch.device(
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

        # 판 내 상태 추적 (train loop 에서 매 판 리셋)
        self._step_count  = 0   # 현재 판의 전체 착수 수
        self._last_opp_move : tuple[int,int] | None = None

    def reset_episode(self):
        """train loop 에서 매 에피소드 시작 전 호출."""
        self._step_count    = 0
        self._last_opp_move = None

    def notify_opp_move(self, move: tuple[int,int]):
        """상대 착수 후 train loop 에서 호출해 상대 마지막 수를 기록."""
        self._last_opp_move = move
        self._step_count   += 1

    # ── 행동 선택 (외부 인터페이스)
    def decide_next_move(self, engine,
                         temperature: float = 1.0,
                         use_mcts   : bool  = False
                         ) -> tuple[int, int] | None:
        board_np = engine.board.board
        cur      = engine.current_player
        n        = self.board_size

        # ── PPO 수 계산
        state   = engine.get_state()
        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value = self.old_net(state_t)

        probs = probs.squeeze(0)
        value = value.item()

        mask_flat = self._build_mask(board_np, cur)
        if not mask_flat.any():
            return None

        probs_masked = probs * mask_flat.float()
        s = probs_masked.sum()
        probs_masked = probs_masked / s if s > 1e-8 \
            else mask_flat.float() / mask_flat.float().sum()

        dist        = Categorical(probs_masked)
        ppo_action  = dist.sample()
        ppo_lp      = dist.log_prob(ppo_action).item()

        # ── MCTS 개입 여부 판단
        final_action = ppo_action.item()
        if self._should_intervene(board_np, cur, n):
            sims = (MCTS_SIMS_LATE
                    if self._step_count >= LATE_GAME_THRESHOLD
                    else MCTS_SIMS_THREAT)
            mcts_action = _shallow_mcts(
                self.old_net, board_np, cur, sims, n, self.device)
            if (mcts_action is not None
                    and mcts_action != ppo_action.item()
                    and mask_flat[mcts_action].item()):
                final_action = mcts_action

        # ── 메모리 저장 (log_prob 은 PPO 분포 기준 유지)
        if not self._eval_mode:
            fa_t = torch.tensor(final_action, device=self.device)
            lp   = torch.log(
                probs_masked[final_action].clamp(min=1e-8)
            ).item()
            self.memory.states.append(state)
            self.memory.actions.append(final_action)
            self.memory.logprobs.append(lp)
            self.memory.values.append(value)

        self._step_count += 1
        self._last_opp_move = None  # 내 수 이후 초기화

        row, col = divmod(final_action, n)
        return (row, col)

    def decide_best_move(self, engine) -> tuple[int, int] | None:
        """평가/대국 시 결정론적 선택 (argmax + MCTS 개입)."""
        board_np = engine.board.board
        cur      = engine.current_player
        n        = self.board_size

        state   = engine.get_state()
        state_t = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, _ = self.old_net(state_t)

        probs     = probs.squeeze(0)
        mask_flat = self._build_mask(board_np, cur)
        if not mask_flat.any():
            return None

        probs  = probs * mask_flat.float()
        action = int(probs.argmax().item())

        # 평가 시에도 후반 MCTS 개입 (sims 줄임)
        if self._should_intervene(board_np, cur, n):
            mcts_action = _shallow_mcts(
                self.old_net, board_np, cur, MCTS_SIMS_LATE, n, self.device)
            if (mcts_action is not None
                    and mask_flat[mcts_action].item()):
                action = mcts_action

        row, col = divmod(action, n)
        return (row, col)

    # ── 개입 조건 판단
    def _should_intervene(self, board_np, player, n) -> bool:
        opp = 3 - player

        # 조건 1: 상대 마지막 수가 3목 이상
        if self._last_opp_move is not None:
            r, c = self._last_opp_move
            if _max_len_at(board_np, r, c, opp, n) >= 3:
                return True

        # 조건 2: 내 돌 중 이미 3목 이상인 줄이 있음 (공격 찬스)
        for r in range(n):
            for c in range(n):
                if board_np[r, c] == player:
                    if _max_len_at(board_np, r, c, player, n) >= 3:
                        return True

        # 조건 3: 게임 후반
        if self._step_count >= LATE_GAME_THRESHOLD:
            return True

        return False

    def _build_mask(self, board_np: np.ndarray,
                    player: int) -> torch.Tensor:
        n    = self.board_size
        mask = (board_np == 0).flatten()

        occupied = np.argwhere(board_np != 0)
        if len(occupied) == 0:
            candidate_mask = np.zeros(n*n, dtype=bool)
            center = n//2
            candidate_mask[center*n+center] = True
        else:
            candidate_mask = np.zeros(n*n, dtype=bool)
            for or_, oc in occupied:
                r0=max(0,int(or_)-2); r1=min(n,int(or_)+3)
                c0=max(0,int(oc)-2);  c1=min(n,int(oc)+3)
                for nr in range(r0,r1):
                    for nc in range(c0,c1):
                        if board_np[nr,nc]==0:
                            candidate_mask[nr*n+nc]=True

        mask = mask & candidate_mask

        if player == 1:
            for idx in np.where(mask)[0]:
                r,c = divmod(int(idx),n)
                if Rules.is_forbidden(board_np,r,c,1):
                    mask[idx] = False

        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    # ── 보상 저장 + 학습
    def store_reward(self, reward: float, done: bool = False):
        if self._eval_mode:
            return
        self.memory.rewards.append(reward * self.REWARD_SCALE)
        self.memory.dones.append(done)

        if done:
            self._ep_count += 1
            if self._ep_count % self.TRAIN_EVERY == 0:
                self.last_loss = self._train()
                self.memory.clear()

    def _train(self) -> float | None:
        T = min(len(self.memory.states), len(self.memory.rewards))
        if T < self.BATCH_SIZE:
            return self.last_loss

        states   = torch.tensor(np.array(self.memory.states[:T]),
                                dtype=torch.float32, device=self.device)
        actions  = torch.tensor(self.memory.actions[:T],
                                dtype=torch.long, device=self.device)
        old_lps  = torch.tensor(self.memory.logprobs[:T],
                                dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(self.memory.values[:T],
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
                mb = idx[start: start+self.BATCH_SIZE]
                if len(mb) < 2: continue
                loss = self._ppo_loss(
                    states[mb], actions[mb],
                    old_lps[mb], advantages[mb], returns[mb])
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item(); count += 1

        self.old_net.load_state_dict(self.net.state_dict())
        return total_loss / max(count, 1)

    def _gae(self, rewards, dones, values) -> torch.Tensor:
        T = len(rewards); advs = torch.zeros(T, device=self.device); gae = 0.0
        for t in reversed(range(T)):
            nv    = values[t+1].item() if t+1 < T else 0.0
            delta = (rewards[t] + self.GAMMA*nv*(1-int(dones[t]))
                     - values[t].item())
            gae   = delta + self.GAMMA*self.GAE_LAMBDA*(1-int(dones[t]))*gae
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
        policy_loss = -torch.min(ratio*advantages, clip_ratio*advantages).mean()
        value_loss  = F.mse_loss(values, returns)

        return policy_loss + self.VALUE_C*value_loss - self.ENTROPY_C*entropy

    # ── 가중치 관리
    def save(self, path: str = 'ppo_p2.pt'):
        torch.save({'net':self.net.state_dict(),
                    'optimizer':self.optimizer.state_dict()}, path)
        print(f'[PPOAgent] 저장 → {path}')

    def load(self, path: str = 'ppo_p2.pt'):
        if not os.path.exists(path):
            print(f'[PPOAgent] {path} 없음 — 랜덤 가중치로 시작'); return
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
        self.old_net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'[PPOAgent] 불러오기 ← {path}')

    def clone_weights(self) -> dict:
        return {k: v.cpu().clone() for k,v in self.net.state_dict().items()}

    def make_champion_agent(self, weights: dict) -> 'PPOAgent':
        champ = PPOAgent(self.board_size, player=2)
        sd    = {k: v.to(self.device) for k,v in weights.items()}
        champ.net.load_state_dict(sd)
        champ.old_net.load_state_dict(sd)
        champ.old_net.eval()
        return champ

    def set_mcts(self, **kwargs):
        pass  # 호환성 유지