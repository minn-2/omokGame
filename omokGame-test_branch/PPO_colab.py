import json
import zipfile
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ai.engine import Engine
from ai.agent  import PPOAgent
from core.Rules import Rules

# ── 경로 설정
CKPT_DIR      = Path('checkpoints')
P2_PATH       = CKPT_DIR / 'ppo_p2.pt'
LOG_PATH      = CKPT_DIR / 'train_log.json'
DRIVE_CKPT    = Path('/content/drive/MyDrive/omok_checkpoints')
KAGGLE_OUT    = Path('/kaggle/working')

# ── 학습 설정
BOARD_SIZE       = 15
TOTAL_EPISODES   = 500000
SAVE_EVERY       = 200
DRIVE_SAVE_EVERY = 1000
EVAL_EVERY       = 500
EVAL_GAMES       = 40        # 반드시 짝수
PROMOTE_WIN_RATE = 0.55
MAX_HALF_MOVES   = BOARD_SIZE * BOARD_SIZE * 2
MAX_INVALID_MOVES = 8

# ── 커리큘럼 설정
# HEURISTIC_UNTIL 에피소드까지는 HeuristicBot 과 대결,
# 이후 셀프플레이로 자동 전환
HEURISTIC_UNTIL  = 150000   # 15만 번까지 휴리스틱 봇


# ──────────────────────────────────────────
# HeuristicBot
# ──────────────────────────────────────────

class HeuristicBot:
    """
    규칙 기반 오목 봇.
    우선순위:
      1. 내 5목 완성
      2. 상대 5목 차단
      3. 내 열린 4목(공격)
      4. 상대 열린 4목 차단
      5. 내 열린 3목
      6. 상대 열린 3목 차단
      7. 중앙 근처 랜덤
    """

    def decide_next_move(self, engine) -> tuple[int, int] | None:
        board  = engine.board.board
        player = engine.current_player
        opp    = 3 - player
        n      = engine.board_size

        # 착수 가능한 칸 (기존 돌 주변 2칸 이내)
        candidates = self._candidates(board, n)
        if not candidates:
            return None

        # 흑돌 금수 필터
        if player == 1:
            candidates = [
                (r, c) for r, c in candidates
                if not Rules.is_forbidden(board, r, c, 1)
            ]
        if not candidates:
            return None

        # 우선순위 순으로 탐색
        for length, target in [
            (5, player),   # 내 5목
            (5, opp),      # 상대 5목 차단
            (4, player),   # 내 4목
            (4, opp),      # 상대 4목 차단
            (3, player),   # 내 3목
            (3, opp),      # 상대 3목 차단
        ]:
            move = self._find_threat(board, candidates, target, length, n)
            if move:
                return move

        # fallback: 중앙 가중 랜덤
        return self._weighted_random(candidates, n)

    # 빈 칸 중 기존 돌 주변 2칸 이내 후보 수집
    def _candidates(self, board, n) -> list[tuple[int, int]]:
        occupied = np.argwhere(board != 0)
        if len(occupied) == 0:
            c = n // 2
            return [(c, c)]
        seen = set()
        for or_, oc in occupied:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = int(or_) + dr, int(oc) + dc
                    if (0 <= nr < n and 0 <= nc < n
                            and board[nr, nc] == 0
                            and (nr, nc) not in seen):
                        seen.add((nr, nc))
        return list(seen)

    # 놓았을 때 length 이상 연속이 되는 칸 탐색
    def _find_threat(self, board, candidates,
                     player, length, n) -> tuple[int, int] | None:
        best      = None
        best_score = -1
        for r, c in candidates:
            board[r, c] = player
            score = self._max_consecutive(board, r, c, player, n)
            board[r, c] = 0
            if score >= length and score > best_score:
                best_score = score
                best       = (r, c)
        return best

    def _max_consecutive(self, board, r, c, player, n) -> int:
        best = 1
        for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
            cnt = 1
            for sign in (1, -1):
                nr, nc = r + dr*sign, c + dc*sign
                while (0 <= nr < n and 0 <= nc < n
                       and board[nr, nc] == player):
                    cnt += 1
                    nr  += dr * sign
                    nc  += dc * sign
            best = max(best, cnt)
        return best

    # 중앙에 가까울수록 높은 가중치로 랜덤 선택
    def _weighted_random(self, candidates,
                         n) -> tuple[int, int] | None:
        if not candidates:
            return None
        center = n // 2
        weights = [
            1.0 / (abs(r - center) + abs(c - center) + 1)
            for r, c in candidates
        ]
        total = sum(weights)
        weights = [w / total for w in weights]
        idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[idx]


# ──────────────────────────────────────────
# 파일 유틸
# ──────────────────────────────────────────

def ensure_dirs():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def save_champion(agent: PPOAgent, episode: int = 0):
    ensure_dirs()
    torch.save({
        'net'      : agent.net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode'  : episode,
    }, P2_PATH)
    print(f'  [♛ CHAMPION] 로컬 저장 완료 (ep {episode:,})')


def save_to_drive(episode: int):
    import shutil
    if KAGGLE_OUT.exists():
        try:
            for p in [P2_PATH, LOG_PATH]:
                if p.exists():
                    shutil.copy(p, KAGGLE_OUT / p.name)
            print(f'  [KAGGLE] 백업 완료 (ep {episode:,})')
        except Exception as e:
            print(f'  [KAGGLE] 백업 실패: {e}')
    elif DRIVE_CKPT.exists():
        try:
            for p in [P2_PATH, LOG_PATH]:
                if p.exists():
                    shutil.copy(p, DRIVE_CKPT / p.name)
            print(f'  [DRIVE] 백업 완료 (ep {episode:,})')
        except Exception as e:
            print(f'  [DRIVE] 백업 실패: {e}')


def load_from_drive():
    import shutil
    kaggle_input = Path('/kaggle/input/datasets/hellocarrot/omokgame/files_omokgame/omokGame-test_branch/checkpoints')
    src_dir = kaggle_input if kaggle_input.exists() else DRIVE_CKPT
    for p in [P2_PATH, LOG_PATH]:
        src = src_dir / p.name
        if src.exists():
            ensure_dirs()
            shutil.copy(src, p)
            print(f'  [LOAD] {p.name} 복사 완료 ← {src}')
        else:
            print(f'  [SKIP] {p.name} 없음')


def load_champion(agent: PPOAgent) -> int:
    if P2_PATH.exists():
        ckpt = torch.load(P2_PATH, map_location=agent.device)
        agent.net.load_state_dict(ckpt['net'])
        agent.old_net.load_state_dict(ckpt['net'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])
        ep = ckpt.get('episode', 0)
        print(f'  [LOAD] 체크포인트 로드 ← {P2_PATH} (ep {ep:,})')
        return ep
    else:
        print(f'  [SKIP] {P2_PATH} 없음 — 랜덤 가중치로 시작')
        return 0


def load_log() -> dict:
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'meta': {
            'board_size': BOARD_SIZE,
            'mode'      : 'Curriculum → Self-Play PPO',
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'updated_at': '',
        },
        'episodes': [],
        'summary' : {
            'total_episodes'  : 0,
            'challenger_wins' : 0,
            'champion_wins'   : 0,
            'draws'           : 0,
            'champion_updates': 0,
        },
    }


def append_log(log: dict, record: dict):
    log['episodes'].append(record)
    s = log['summary']
    s['total_episodes'] += 1
    w = record.get('challenger_won')
    if w is True   : s['challenger_wins'] += 1
    elif w is False: s['champion_wins']   += 1
    else           : s['draws']           += 1
    if record.get('champion_updated'):
        s['champion_updates'] += 1
    log['meta']['updated_at'] = datetime.now().isoformat(timespec='seconds')


def save_log(log: dict):
    ensure_dirs()
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def export_zip(out_dir: str = '.') -> str:
    ensure_dirs()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zp = Path(out_dir) / f'omok_p2_{ts}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in [P2_PATH, LOG_PATH]:
            if p.exists():
                zf.write(p, arcname=p.name)
                print(f'  [ZIP] {p.name}')
    print(f'[EXPORT] → {zp}')
    return str(zp)


# ──────────────────────────────────────────
# 보상 함수
# ──────────────────────────────────────────

def _max_consecutive(board, r, c, player, n) -> int:
    """(r,c) 돌 기준 4방향 최대 연속 길이."""
    best = 1
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt = 1
        for sign in (1, -1):
            nr, nc = r + dr*sign, c + dc*sign
            while (0 <= nr < n and 0 <= nc < n
                   and board[nr, nc] == player):
                cnt += 1
                nr  += dr * sign
                nc  += dc * sign
        best = max(best, cnt)
    return best


def _board_max_consecutive(board, player, n) -> int:
    """보드 전체에서 player 의 최대 연속 길이."""
    best = 0
    for r in range(n):
        for c in range(n):
            if board[r, c] == player:
                v = _max_consecutive(board, r, c, player, n)
                if v > best:
                    best = v
    return best


def shaped_reward(engine: Engine, move: tuple[int, int],
                  player: int) -> float:
    """
    make_move() 직후 호출.
    방금 놓은 move 좌표 기준으로 보상 계산 → 중복 보상 없음.
    """
    board = engine.board.board
    opp   = 3 - player
    n     = engine.board_size

    if engine.is_over:
        if engine.winner == player: return  50.0
        if engine.winner == opp   : return -45.0
        return 0.0

    r, c   = move
    reward = 0.2  # 생존 보너스

    # 내 연속 길이 (방금 놓은 돌 기준)
    my_len = _max_consecutive(board, r, c, player, n)
    if   my_len >= 4: reward += 8.0
    elif my_len == 3: reward += 2.0

    # 상대 현재 최대 연속 길이 (방치 페널티)
    opp_max = _board_max_consecutive(board, opp, n)
    if   opp_max >= 4: reward -= 12.0
    elif opp_max == 3: reward -=  3.0

    return float(reward)


def safe_make_move(env: Engine, move) -> bool:
    if move is None:
        return False
    try:
        return bool(env.make_move(*move))
    except Exception:
        return False


# ──────────────────────────────────────────
# Champion 평가
# ──────────────────────────────────────────

def evaluate(challenger: PPOAgent,
             champ_weights: dict,
             n_games: int = EVAL_GAMES) -> dict:
    assert n_games % 2 == 0
    half  = n_games // 2
    champ = challenger.make_champion_agent(champ_weights)

    # 평가 전 메모리 백업
    mem_bak = (
        list(challenger.memory.states),
        list(challenger.memory.actions),
        list(challenger.memory.logprobs),
        list(challenger.memory.values),
        list(challenger.memory.rewards),
        list(challenger.memory.dones),
    )
    challenger._eval_mode = True

    black_wins = white_wins = 0
    for g in range(n_games):
        env      = Engine(BOARD_SIZE)
        env.reset()
        ch_color = 1 if g < half else 2
        agents   = {ch_color: challenger, 3-ch_color: champ}

        step = invalid = 0
        while not env.is_over and step < MAX_HALF_MOVES:
            cur  = env.current_player
            move = agents[cur].decide_next_move(env)
            if not safe_make_move(env, move):
                invalid += 1
                if invalid >= MAX_INVALID_MOVES: break
                continue
            invalid = 0
            step   += 1

        if env.winner == ch_color:
            if ch_color == 1: black_wins += 1
            else            : white_wins += 1

    challenger._eval_mode = False
    # 메모리 복원
    (challenger.memory.states,
     challenger.memory.actions,
     challenger.memory.logprobs,
     challenger.memory.values,
     challenger.memory.rewards,
     challenger.memory.dones) = mem_bak

    return {
        'total': (black_wins + white_wins) / n_games,
        'black': black_wins / half,
        'white': white_wins / half,
    }


# ──────────────────────────────────────────
# 터미널 출력
# ──────────────────────────────────────────

def print_stats(ep: int, stats: dict, eval_result: dict | None,
                phase: str):
    bar_len  = 30
    progress = int(ep / TOTAL_EPISODES * bar_len)
    bar      = '█' * progress + '░' * (bar_len - progress)

    chw  = stats['challenger_wins']
    cpw  = stats['champion_wins']
    drw  = stats['draws']
    tot  = max(chw + cpw + drw, 1)
    loss = stats.get('loss')

    print(f'\n{"─"*60}')
    print(f'  [{phase}]  에피소드 {ep:>7,} / {TOTAL_EPISODES:,}')
    print(f'  [{bar}] {ep / TOTAL_EPISODES * 100:.1f}%')
    if isinstance(loss, float):
        print(f'  스텝: {stats["steps"]:,}   보상: {stats["ep_reward"]:+.2f}'
              f'   Loss: {loss:.6f}')
    else:
        print(f'  스텝: {stats["steps"]:,}   보상: {stats["ep_reward"]:+.2f}'
              f'   Loss: —')
    print(f'  Challenger {"흑" if stats["ch_color"] == 1 else "백"}돌')
    print(f'  승패  Ch {chw:,}({chw/tot*100:.1f}%)'
          f'  Opp {cpw:,}({cpw/tot*100:.1f}%)'
          f'  무 {drw:,}({drw/tot*100:.1f}%)')

    if eval_result:
        tag = '♛ CHAMPION 갱신!' if stats.get('champion_updated') else '─ 유지'
        print(f'  ┌ 평가 결과 [{tag}]')
        print(f'  │  전체: {eval_result["total"]*100:.1f}%'
              f'  흑돌: {eval_result["black"]*100:.1f}%'
              f'  백돌: {eval_result["white"]*100:.1f}%')
        print(f'  └  역대 최고: {stats["best_eval_wr"]*100:.1f}%'
              f'  총 갱신: {stats["champion_updates"]}회')
    print(f'{"─"*60}')


# ──────────────────────────────────────────
# 메인 학습 루프
# ──────────────────────────────────────────

def train(resume: bool = False):
    ensure_dirs()

    challenger = PPOAgent(BOARD_SIZE, player=2)
    challenger._eval_mode = False

    start_ep = 0
    if resume:
        print('[RESUME] 체크포인트 복사 중...')
        load_from_drive()
        start_ep = load_champion(challenger)

    if not P2_PATH.exists():
        save_champion(challenger, episode=0)
        save_to_drive(0)

    champ_weights = challenger.clone_weights()
    log           = load_log()
    summ          = log['summary']
    best_eval_wr  = 0.0

    heuristic_bot = HeuristicBot()

    stats: dict = {
        'episode'         : start_ep,
        'steps'           : 0,
        'challenger_wins' : summ['challenger_wins'],
        'champion_wins'   : summ['champion_wins'],
        'draws'           : summ['draws'],
        'ep_reward'       : 0.0,
        'loss'            : None,
        'eval_win_rate'   : None,
        'best_eval_wr'    : 0.0,
        'champion_updated': False,
        'champion_updates': summ['champion_updates'],
        'ch_color'        : 1,
    }

    phase_boundary = HEURISTIC_UNTIL
    print(f'\n[TRAIN] Curriculum → Self-Play PPO')
    print(f'  장치           : {challenger.device}')
    print(f'  시작 에피소드  : {start_ep + 1:,}')
    print(f'  총 에피소드    : {TOTAL_EPISODES:,}')
    print(f'  ── Phase 1: ep 1 ~ {phase_boundary:,}  (vs HeuristicBot)')
    print(f'  ── Phase 2: ep {phase_boundary+1:,} ~ {TOTAL_EPISODES:,}  (Self-Play)')
    print(f'  평가 주기      : {EVAL_EVERY}ep  ({EVAL_GAMES}판)')
    print(f'  champion 기준  : 전체 승률 {PROMOTE_WIN_RATE*100:.0f}% 이상 + 역대 최고\n')

    last_eval: dict | None = None

    for ep in range(start_ep + 1, TOTAL_EPISODES + 1):

        # ── Phase 결정
        use_heuristic = (ep <= phase_boundary)
        phase_str     = 'Phase1 HeuristicBot' if use_heuristic else 'Phase2 SelfPlay'

        # Phase 전환 알림 (딱 한 번)
        if ep == phase_boundary + 1:
            print(f'\n{"="*60}')
            print(f'  [전환] Phase 2 시작 — Self-Play (ep {ep:,})')
            print(f'  champion 가중치를 현재 challenger 와 동기화')
            print(f'{"="*60}\n')
            # champion을 현재 challenger 로 리셋해서 셀프플레이 시작
            champ_weights = challenger.clone_weights()
            save_champion(challenger, episode=ep)

        ch_color    = 1 if ep % 2 == 1 else 2
        champ_color = 3 - ch_color

        # ── 상대 결정
        if use_heuristic:
            # Phase 1: challenger vs HeuristicBot
            agents = {
                ch_color   : challenger,
                champ_color: heuristic_bot,
            }
        else:
            # Phase 2: challenger vs champion (Self-Play)
            champ_agent = challenger.make_champion_agent(champ_weights)
            agents      = {
                ch_color   : challenger,
                champ_color: champ_agent,
            }

        env = Engine(BOARD_SIZE)
        env.reset()
        challenger.memory.clear()

        ep_reward        = 0.0
        step             = 0
        champion_updated = False

        # ── 한 판 진행
        invalid = 0
        while not env.is_over and step < MAX_HALF_MOVES:
            cur   = env.current_player
            agent = agents[cur]
            move  = agent.decide_next_move(env)
            if not safe_make_move(env, move):
                invalid += 1
                if invalid >= MAX_INVALID_MOVES:
                    break
                continue
            invalid = 0
            step   += 1

            if cur == ch_color:
                r = shaped_reward(env, move, ch_color)
                ep_reward += r
                challenger.store_reward(r, env.is_over)
            elif env.is_over and cur != ch_color:
                # 상대 마지막 수로 게임 종료 → 패배 보상 전달
                r = shaped_reward(env, move, ch_color)
                ep_reward += r
                challenger.store_reward(r, True)

        # ── 승패 집계
        winner = env.winner
        if winner == ch_color:
            challenger_won = True;  summ['challenger_wins'] += 1
        elif winner == champ_color:
            challenger_won = False; summ['champion_wins']   += 1
        else:
            challenger_won = None;  summ['draws']           += 1

        # ── champion 평가 (Phase 2 에서만 의미 있음)
        if ep % EVAL_EVERY == 0 and not use_heuristic:
            print(f'\n[EVAL] ep {ep:,} — {EVAL_GAMES}판 평가 중...')
            wr        = evaluate(challenger, champ_weights, EVAL_GAMES)
            last_eval = wr
            print(f'  전체: {wr["total"]*100:.1f}%'
                  f'  흑돌: {wr["black"]*100:.1f}%'
                  f'  백돌: {wr["white"]*100:.1f}%')

            if wr['total'] >= PROMOTE_WIN_RATE and wr['total'] > best_eval_wr:
                best_eval_wr     = wr['total']
                save_champion(challenger, episode=ep)
                save_to_drive(ep)
                champ_weights    = challenger.clone_weights()
                champion_updated = True
                stats['champion_updates'] += 1
                summ['champion_updates']  += 1
                print(f'  [♛] Champion 갱신! (총 {summ["champion_updates"]}회)')
            else:
                print(f'  [─] 유지  (역대 최고: {best_eval_wr*100:.1f}%)')

            stats['eval_win_rate']    = wr
            stats['best_eval_wr']     = best_eval_wr
            stats['champion_updated'] = champion_updated

        # Phase 1 에서는 SAVE_EVERY 마다 저장만 (평가 없음)
        elif ep % EVAL_EVERY == 0 and use_heuristic:
            save_champion(challenger, episode=ep)
            save_log(log)

        if ep % SAVE_EVERY == 0:
            save_champion(challenger, episode=ep)
            save_log(log)

        if ep % DRIVE_SAVE_EVERY == 0:
            save_to_drive(ep)
            save_log(log)

        # ── 로그
        record = {
            'episode'         : ep,
            'phase'           : phase_str,
            'ch_color'        : ch_color,
            'winner'          : int(winner) if winner is not None else 0,
            'challenger_won'  : challenger_won,
            'steps'           : step,
            'ep_reward'       : round(ep_reward, 4),
            'loss'            : challenger.last_loss,
            'champion_updated': champion_updated,
            'timestamp'       : datetime.now().isoformat(timespec='seconds'),
        }
        append_log(log, record)

        stats.update({
            'episode'        : ep,
            'steps'          : step,
            'ep_reward'      : ep_reward,
            'challenger_wins': summ['challenger_wins'],
            'champion_wins'  : summ['champion_wins'],
            'draws'          : summ['draws'],
            'loss'           : challenger.last_loss,
            'ch_color'       : ch_color,
        })

        if ep % 100 == 0:
            print_stats(ep, stats,
                        last_eval if ep % EVAL_EVERY == 0 else None,
                        phase_str)
            last_eval = None

    save_champion(challenger, episode=ep)
    save_to_drive(ep)
    save_log(log)
    print('\n[DONE] 학습 종료 — 최종 저장 완료')


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='오목 Curriculum→Self-Play PPO')
    parser.add_argument('--resume', action='store_true',
                        help='체크포인트를 불러와 이어서 학습')
    parser.add_argument('--export', action='store_true',
                        help='ZIP 내보내기만 실행')
    args = parser.parse_args()

    if args.export:
        export_zip()
    else:
        train(resume=args.resume)