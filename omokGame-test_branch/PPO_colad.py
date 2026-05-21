import json
import zipfile
import argparse
from datetime import datetime
from pathlib import Path

import torch

from ai.engine import Engine
from ai.agent  import PPOAgent
from core.Rules import Rules

# ── 경로 설정
CKPT_DIR      = Path('checkpoints')               # 로컬 체크포인트 폴더
P2_PATH       = CKPT_DIR / 'ppo_p2.pt'            # 모델 가중치 파일
LOG_PATH      = CKPT_DIR / 'train_log.json'        # 학습 로그 파일
DRIVE_CKPT    = Path('/content/drive/MyDrive/omok_checkpoints')  # Drive 백업 경로

# ── 학습 설정
BOARD_SIZE       = 15
TOTAL_EPISODES   = 1000000
SAVE_EVERY       = 200       # 로컬 저장 주기
DRIVE_SAVE_EVERY = 1000      # Drive 백업 주기 (세션 끊김 대비)
EVAL_EVERY       = 200       # champion 평가 주기
EVAL_GAMES       = 60        # 평가 대결 수 (흑/백 각 30판, 반드시 짝수)
PROMOTE_WIN_RATE = 0.55      # champion 갱신 기준 승률


# ──────────────────────────────────────────
# 파일 유틸
# ──────────────────────────────────────────

def ensure_dirs():
    """로컬 및 Drive 체크포인트 폴더 생성."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_CKPT.mkdir(parents=True, exist_ok=True)


def save_champion(agent: PPOAgent, episode: int = 0):
    """현재 모델을 로컬 ppo_p2.pt 에 저장."""
    ensure_dirs()
    torch.save({
        'net'      : agent.net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode'  : episode,
    }, P2_PATH)
    print(f'  [♛ CHAMPION] 로컬 저장 완료 (ep {episode:,})')


def save_to_drive(episode: int):
    """로컬 체크포인트를 Google Drive 에 백업."""
    try:
        import shutil
        ensure_dirs()
        for p in [P2_PATH, LOG_PATH]:
            if p.exists():
                shutil.copy(p, DRIVE_CKPT / p.name)
        print(f'  [DRIVE] 백업 완료 (ep {episode:,}) → {DRIVE_CKPT}')
    except Exception as e:
        print(f'  [DRIVE] 백업 실패: {e}')


def load_from_drive():
    """Drive 체크포인트를 로컬로 복사 (resume 시 호출)."""
    try:
        import shutil
        for p in [P2_PATH, LOG_PATH]:
            src = DRIVE_CKPT / p.name
            if src.exists():
                ensure_dirs()
                shutil.copy(src, p)
                print(f'  [DRIVE→LOCAL] {p.name} 복사 완료')
            else:
                print(f'  [DRIVE→LOCAL] {p.name} 없음 — 스킵')
    except Exception as e:
        print(f'  [DRIVE→LOCAL] 실패: {e}')


def load_champion(agent: PPOAgent) -> int:
    """로컬 ppo_p2.pt 를 에이전트에 로드. 저장된 에피소드 번호 반환."""
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
    """기존 학습 로그 불러오기. 없으면 새로 생성."""
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'meta': {
            'board_size': BOARD_SIZE,
            'mode'      : 'Self-Play PPO',
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
    """에피소드 결과를 로그에 추가."""
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
    """학습 로그를 JSON 파일로 저장."""
    ensure_dirs()
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def export_zip(out_dir: str = '.') -> str:
    """체크포인트와 로그를 ZIP 으로 묶어서 내보내기."""
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

def shaped_reward(engine: Engine, player: int) -> float:
    """
    승패 보상 + 패턴 기반 중간 보상.
    - 승리: +50 / 패배: -50
    - 4목 형성: +8 / 상대 4목: -15
    - 3목 형성: +3 / 상대 3목: -5
    실제 agent.store_reward() 에서 REWARD_SCALE(0.02) 로 스케일 조정됨.
    """
    board = engine.board.board
    opp   = 3 - player
    if engine.is_over:
        if engine.winner == player: return  50.0
        if engine.winner == opp   : return -50.0
        return 0.0
    r = 0.0
    if Rules.check_patterns(board, player, 4): r += 8.0
    if Rules.check_patterns(board, opp,    4): r -= 15.0
    if Rules.check_patterns(board, player, 3): r += 3.0
    if Rules.check_patterns(board, opp,    3): r -= 5.0
    return float(r)


# ──────────────────────────────────────────
# Champion 평가 — 흑/백 승률 분리
# ──────────────────────────────────────────

def evaluate(challenger: PPOAgent,
             champ_weights: dict,
             n_games: int = EVAL_GAMES) -> dict:
    """
    Challenger vs Champion 평가전.
    전반 n_games//2 판: challenger = 흑돌(1)
    후반 n_games//2 판: challenger = 백돌(2)
    → 흑/백 승률을 분리하여 반환.

    Returns:
        {'total': float, 'black': float, 'white': float}
    """
    assert n_games % 2 == 0, "EVAL_GAMES 는 짝수여야 합니다"
    half  = n_games // 2
    champ = challenger.make_champion_agent(champ_weights)

    black_wins = 0  # challenger 가 흑돌일 때 승리 횟수
    white_wins = 0  # challenger 가 백돌일 때 승리 횟수

    for g in range(n_games):
        env = Engine(BOARD_SIZE)
        env.reset()

        # 전반: challenger = 흑(1) / 후반: challenger = 백(2)
        if g < half:
            ch_color = 1
            agents   = {1: challenger, 2: champ}
        else:
            ch_color = 2
            agents   = {1: champ, 2: challenger}

        while not env.is_over:
            cur  = env.current_player
            move = agents[cur].decide_next_move(env)
            if move is None:
                break
            env.make_move(*move)

        if env.winner == ch_color:
            if ch_color == 1: black_wins += 1
            else            : white_wins += 1

    # 평가 후 메모리 초기화 (평가 데이터가 학습에 섞이지 않도록)
    challenger.memory.clear()

    return {
        'total': (black_wins + white_wins) / n_games,
        'black': black_wins / half,
        'white': white_wins / half,
    }


# ──────────────────────────────────────────
# 터미널 출력 (Pygame 대신 사용)
# ──────────────────────────────────────────

def print_stats(ep: int, stats: dict, eval_result: dict | None = None):
    """100 에피소드마다 학습 현황을 터미널에 출력."""
    bar_len  = 30
    progress = int(ep / TOTAL_EPISODES * bar_len)
    bar      = '█' * progress + '░' * (bar_len - progress)

    chw = stats['challenger_wins']
    cpw = stats['champion_wins']
    drw = stats['draws']
    tot = max(chw + cpw + drw, 1)
    loss = stats.get('loss')

    print(f'\n{"─"*58}')
    print(f'  에피소드  {ep:>7,} / {TOTAL_EPISODES:,}')
    print(f'  [{bar}] {ep / TOTAL_EPISODES * 100:.1f}%')
    if isinstance(loss, float):
        print(f'  스텝: {stats["steps"]:,}   보상: {stats["ep_reward"]:+.2f}'
              f'   Loss: {loss:.6f}')
    else:
        print(f'  스텝: {stats["steps"]:,}   보상: {stats["ep_reward"]:+.2f}'
              f'   Loss: —')
    print(f'  Challenger {"흑" if stats["ch_color"] == 1 else "백"}돌')
    print(f'  승패  Ch {chw:,}({chw/tot*100:.1f}%)'
          f'  Champ {cpw:,}({cpw/tot*100:.1f}%)'
          f'  무 {drw:,}({drw/tot*100:.1f}%)')

    if eval_result:
        tag = '♛ CHAMPION 갱신!' if stats.get('champion_updated') else '─ 유지'
        print(f'  ┌ 평가 결과 [{tag}]')
        print(f'  │  전체: {eval_result["total"]*100:.1f}%'
              f'  흑돌: {eval_result["black"]*100:.1f}%'
              f'  백돌: {eval_result["white"]*100:.1f}%')
        print(f'  └  역대 최고: {stats["best_eval_wr"]*100:.1f}%'
              f'  총 갱신: {stats["champion_updates"]}회')
    print(f'{"─"*58}')


# ──────────────────────────────────────────
# 메인 학습 루프
# ──────────────────────────────────────────

def train(resume: bool = False):
    ensure_dirs()

    # 에이전트 생성
    challenger = PPOAgent(BOARD_SIZE, player=2)

    # resume: Drive → 로컬 복사 후 가중치 로드
    start_ep = 0
    if resume:
        print('[RESUME] Drive에서 체크포인트 복사 중...')
        load_from_drive()
        start_ep = load_champion(challenger)

    # 첫 실행이면 랜덤 가중치를 champion 으로 저장
    if not P2_PATH.exists():
        save_champion(challenger, episode=0)
        save_to_drive(0)

    # champion 가중치 스냅샷 (challenger 가 이길 때만 갱신)
    champ_weights = challenger.clone_weights()

    log          = load_log()
    summ         = log['summary']
    best_eval_wr = 0.0

    # 통계 딕셔너리 초기화
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

    print(f'\n[TRAIN] Self-Play PPO 학습 시작')
    print(f'  장치           : {challenger.device}')
    print(f'  시작 에피소드  : {start_ep + 1:,}')
    print(f'  총 에피소드    : {TOTAL_EPISODES:,}')
    print(f'  평가 주기      : {EVAL_EVERY}ep  ({EVAL_GAMES}판, 흑/백 각 {EVAL_GAMES//2}판)')
    print(f'  Drive 백업주기 : {DRIVE_SAVE_EVERY}ep마다')
    print(f'  champion 기준  : 전체 승률 {PROMOTE_WIN_RATE*100:.0f}% 이상 + 역대 최고\n')

    last_eval: dict | None = None

    for ep in range(start_ep + 1, TOTAL_EPISODES + 1):

        # 홀수 에피소드: challenger = 흑(1) / 짝수: challenger = 백(2)
        ch_color    = 1 if ep % 2 == 1 else 2
        champ_color = 3 - ch_color

        # champion 스냅샷으로 상대 에이전트 생성
        champ_agent = challenger.make_champion_agent(champ_weights)
        agents      = {ch_color: challenger, champ_color: champ_agent}

        env = Engine(BOARD_SIZE)
        env.reset()
        challenger.memory.clear()

        ep_reward        = 0.0
        step             = 0
        champion_updated = False

        # ── 한 판 진행
        while not env.is_over:
            cur   = env.current_player
            agent = agents[cur]
            move  = agent.decide_next_move(env)
            if move is None:
                break
            env.make_move(*move)
            step += 1

            # challenger 차례일 때만 보상 기록
            if cur == ch_color:
                r = shaped_reward(env, ch_color)
                ep_reward += r
                # store_reward 내부에서 TRAIN_EVERY(10)판마다 PPO 학습 실행
                challenger.store_reward(r, env.is_over)

        # ── 승패 집계
        winner = env.winner
        if winner == ch_color:
            challenger_won = True;  summ['challenger_wins'] += 1
        elif winner == champ_color:
            challenger_won = False; summ['champion_wins']   += 1
        else:
            challenger_won = None;  summ['draws']           += 1

        # ── champion 평가 (EVAL_EVERY 마다)
        if ep % EVAL_EVERY == 0:
            print(f'\n[EVAL] ep {ep:,} — {EVAL_GAMES}판 평가 중'
                  f' (흑 {EVAL_GAMES//2}판 / 백 {EVAL_GAMES//2}판)...')
            wr        = evaluate(challenger, champ_weights, EVAL_GAMES)
            last_eval = wr
            print(f'  전체: {wr["total"]*100:.1f}%'
                  f'  흑돌: {wr["black"]*100:.1f}%'
                  f'  백돌: {wr["white"]*100:.1f}%'
                  f'  (기준 {PROMOTE_WIN_RATE*100:.0f}%)')

            # 전체 승률이 기준 이상 + 역대 최고일 때만 champion 갱신
            if wr['total'] >= PROMOTE_WIN_RATE and wr['total'] > best_eval_wr:
                best_eval_wr     = wr['total']
                save_champion(challenger, episode=ep)
                save_to_drive(ep)                    # 갱신 즉시 Drive 백업
                champ_weights    = challenger.clone_weights()
                champion_updated = True
                stats['champion_updates'] += 1
                summ['champion_updates']  += 1
                print(f'  [♛] Champion 갱신! (총 {summ["champion_updates"]}회)')
            else:
                print(f'  [─] Champion 유지  (역대 최고: {best_eval_wr*100:.1f}%)')

            stats['eval_win_rate']    = wr
            stats['best_eval_wr']     = best_eval_wr
            stats['champion_updated'] = champion_updated

        # ── 로컬 정기 저장
        if ep % SAVE_EVERY == 0:
            save_champion(challenger, episode=ep)
            save_log(log)

        # ── Drive 정기 백업 (세션 끊김 대비)
        if ep % DRIVE_SAVE_EVERY == 0:
            save_to_drive(ep)
            save_log(log)

        # ── 로그 기록
        record = {
            'episode'         : ep,
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

        # ── stats 업데이트
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

        # ── 터미널 출력 (100ep 마다)
        if ep % 100 == 0:
            print_stats(
                ep, stats,
                last_eval if ep % EVAL_EVERY == 0 else None)
            last_eval = None

    # ── 학습 종료 후 최종 저장
    save_champion(challenger, episode=ep)
    save_to_drive(ep)
    save_log(log)
    print('\n[DONE] 학습 종료 — 최종 저장 완료')


# ──────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='오목 Self-Play PPO (Colab용)')
    parser.add_argument('--resume', action='store_true',
                        help='Drive에서 체크포인트를 불러와 이어서 학습')
    parser.add_argument('--export', action='store_true',
                        help='ZIP 내보내기만 실행하고 종료')
    args = parser.parse_args()

    if args.export:
        export_zip()
    else:
        train(resume=args.resume)