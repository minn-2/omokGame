import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core.Rules import Rules
from ai.model  import PPOModel

SEARCH_SIMS_THREAT      = 60
SEARCH_SIMS_LATE        = 80
LATE_GAME_THRESHOLD   = 15   
C_EXPLORE                = 1.5

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

# 룰 기반 유틸
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

def _max_len_at(board, r, c, player, n):
    best = 1
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt = 1
        for s in (1,-1):
            nr, nc = r+dr*s, c+dc*s
            while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                cnt+=1; nr+=dr*s; nc+=dc*s
        best = max(best, cnt)
    return best

def _count_open_ends(board, r, c, player, n):
    score = 0
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt = 1
        ends = []
        for s in (1,-1):
            nr, nc = r+dr*s, c+dc*s
            while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                cnt+=1; nr+=dr*s; nc+=dc*s
            if 0<=nr<n and 0<=nc<n and board[nr,nc]==0:
                ends.append(1)
            else:
                ends.append(0)
        if cnt >= 3 and sum(ends) >= 1:
            score += cnt * sum(ends)
    return score

def _valid_mask(board, player, n):
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

# 인라인 트리 탐색 (Tree Search)
class _SearchNode:
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

def _tree_search(net, board_np, player, n_sims, n, device):
    root = _SearchNode(None, -1, 1.0)
    root.n = 1

    def _expand(node, b, p):
        mask = _valid_mask(b, p, n)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            node.expanded = True; return

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

        top_k = 15
        if len(idxs) > top_k:
            top_idxs = np.argsort(pr)[::-1][:top_k]
            mask2 = np.zeros(n*n, dtype=bool)
            mask2[top_idxs] = True
            mask = mask & mask2
            idxs = np.where(mask)[0]

        for idx in idxs:
            node.children[int(idx)] = _SearchNode(node, int(idx), float(pr[idx]))
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

        while node.expanded and node.children:
            node = node.best_child(C_EXPLORE)
            r,c  = divmod(node.action, n)
            b[r,c] = p; p = 3-p

        if node.action >= 0 and _check_win(b, node.action, 3-p, n):
            v = -1.0
        elif not (b==0).any():
            v = 0.0
        else:
            if not node.expanded:
                _expand(node, b, p)
                if node.children:
                    node = node.best_child(C_EXPLORE)
                    r,c  = divmod(node.action, n)
                    b[r,c] = p; p = 3-p
            v = _evaluate(b, p)

        cur = node
        while cur is not None:
            cur.n += 1; cur.w += v; v = -v; cur = cur.parent

    if not root.children:
        return None
    best = max(root.children.values(), key=lambda x: x.n)
    return best.action

# PPOAgent
class PPOAgent:
    LR           = 3e-4
    GAMMA        = 0.99
    GAE_LAMBDA   = 0.95
    CLIP_EPS     = 0.2
    ENTROPY_C    = 0.01
    VALUE_C      = 0.5
    EPOCHS       = 4
    BATCH_SIZE   = 48
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

        self._step_count  = 0
        self._last_opp_move : tuple[int,int] | None = None

    def reset_episode(self):
        self._step_count    = 0
        self._last_opp_move = None

    def notify_opp_move(self, move: tuple[int,int]):
        self._last_opp_move = move
        self._step_count   += 1

    # 룰 기반 선처리 
    def _rule_based_move(self, board_np: np.ndarray,
                         player: int, n: int
                         ) -> tuple[int,int] | None:
        opp       = 3 - player
        my_mask   = _valid_mask(board_np, player, n)
        opp_mask  = _valid_mask(board_np, opp, n)
        my_idxs   = np.where(my_mask)[0]
        opp_idxs  = np.where(opp_mask)[0]

        # 1순위: 내 즉시 승리
        for idx in my_idxs:
            r, c = divmod(int(idx), n)
            board_np[r, c] = player
            win = _check_win(board_np, idx, player, n)
            board_np[r, c] = 0
            if win:
                return (r, c)

        # 2순위: 상대 즉시 승리 차단
        for idx in opp_idxs:
            r, c = divmod(int(idx), n)
            board_np[r, c] = opp
            win = _check_win(board_np, idx, opp, n)
            board_np[r, c] = 0
            if win and my_mask[idx]:
                return (r, c)

        # 3순위: 내 열린 4목 완성 (상대가 막기 전에)
        best_atk, best_atk_score = None, 0
        for idx in my_idxs:
            r, c = divmod(int(idx), n)
            board_np[r, c] = player
            length = _max_len_at(board_np, r, c, player, n)
            opens  = _count_open_ends(board_np, r, c, player, n)
            board_np[r, c] = 0
            if length >= 4 and opens > best_atk_score:
                best_atk_score = opens
                best_atk = (r, c)
        if best_atk is not None:
            return best_atk

        # 4순위: 상대 열린 4목 차단
        best_def, best_def_score = None, 0
        for idx in opp_idxs:
            r, c = divmod(int(idx), n)
            board_np[r, c] = opp
            length = _max_len_at(board_np, r, c, opp, n)
            opens  = _count_open_ends(board_np, r, c, opp, n)
            board_np[r, c] = 0
            if length >= 4 and opens > best_def_score and my_mask[idx]:
                best_def_score = opens
                best_def = (r, c)
        if best_def is not None:
            return best_def

        # 5순위: 상대 4목(닫힌 쪽 포함) 차단
        for idx in opp_idxs:
            r, c = divmod(int(idx), n)
            board_np[r, c] = opp
            length = _max_len_at(board_np, r, c, opp, n)
            board_np[r, c] = 0
            if length >= 4 and my_mask[idx]:
                return (r, c)

        return None  # 룰로 결정 못 하면 모델에 위임

    def _should_intervene(self, board_np, player, n) -> bool:
        opp = 3 - player

        if self._last_opp_move is not None:
            r, c = self._last_opp_move
            if _max_len_at(board_np, r, c, opp, n) >= 2:
                return True

        for r in range(n):
            for c in range(n):
                if board_np[r, c] == player:
                    if _max_len_at(board_np, r, c, player, n) >= 3:  # ← 한 단계 더 들여쓰기
                        return True

        return False
    
    def decide_next_move(self, engine,
                         temperature: float = 1.0,
                         use_search   : bool  = False
                         ) -> tuple[int, int] | None:
        board_np = engine.board.board
        cur      = engine.current_player
        n        = self.board_size

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

        dist       = Categorical(probs_masked)
        ppo_action = dist.sample()

        final_action = ppo_action.item()

        if not self._eval_mode:
            lp = torch.log(
                probs_masked[final_action].clamp(min=1e-8)
            ).item()
            self.memory.states.append(state)
            self.memory.actions.append(final_action)
            self.memory.logprobs.append(lp)
            self.memory.values.append(value)

        self._step_count += 1
        self._last_opp_move = None

        row, col = divmod(final_action, n)
        return (row, col)

    def decide_best_move(self, engine,
                         no_search: bool = False
                         ) -> tuple[int, int] | None:
        board_np = engine.board.board
        cur      = engine.current_player
        n        = self.board_size

        # 1) 룰 기반 선처리
        rule_move = self._rule_based_move(board_np, cur, n)
        if rule_move is not None:
            self._step_count += 1
            self._last_opp_move = None
            return rule_move

        # 2) 모델 policy 준비
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

        probs_masked = probs * mask_flat.float()
        action = int(probs_masked.argmax().item())

        if not no_search and self._should_intervene(board_np, cur, n):
            n_sims = (SEARCH_SIMS_THREAT
                      if self._step_count < LATE_GAME_THRESHOLD
                      else SEARCH_SIMS_LATE)
            search_action = _tree_search(
                self.old_net, board_np, cur, n_sims, n, self.device)
            if (search_action is not None
                    and mask_flat[search_action].item()):
                action = search_action

        self._step_count += 1
        self._last_opp_move = None

        row, col = divmod(action, n)
        return (row, col)
    
    # 마스크 빌드
    def _build_mask(self, board_np: np.ndarray,
                    player: int) -> torch.Tensor:
        n        = self.board_size
        empty    = (board_np == 0)
        occupied = (board_np != 0)

        if not occupied.any():
            candidate_mask = np.zeros((n, n), dtype=bool)
            candidate_mask[n//2, n//2] = True
        else:
            # 주변 2칸 이내에 돌이 있는 빈 칸을 후보로 추림
            # numpy 패딩 후 5×5 슬라이딩 윈도우 max → scipy 없이 동일 결과
            pad      = np.pad(occupied, 2, mode='constant', constant_values=False)
            neighbor = np.zeros((n, n), dtype=bool)
            for dr in range(5):
                for dc in range(5):
                    neighbor |= pad[dr:dr+n, dc:dc+n]
            candidate_mask = neighbor & empty

        mask = empty & candidate_mask
        flat = mask.flatten()
        return torch.tensor(flat, dtype=torch.bool, device=self.device)

    # 보상 저장 + 학습
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

    # 가중치 관리
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

    def set_search(self, **kwargs):
        pass