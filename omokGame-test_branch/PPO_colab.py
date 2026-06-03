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

CKPT_DIR      = Path('checkpoints')
P2_PATH       = CKPT_DIR / 'ppo_p2.pt'
LOG_PATH      = CKPT_DIR / 'train_log.json'
DRIVE_CKPT    = Path('/content/drive/MyDrive/omok_checkpoints')
KAGGLE_OUT    = Path('/kaggle/working')

BOARD_SIZE        = 15
TOTAL_EPISODES    = 1_000_000
SAVE_EVERY        = 500
DRIVE_SAVE_EVERY  = 2000
EVAL_EVERY        = 1000
EVAL_GAMES        = 20
PROMOTE_WIN_RATE  = 0.56 
MAX_HALF_MOVES    = BOARD_SIZE * BOARD_SIZE * 2
MAX_INVALID_MOVES = 8
HEURISTIC_UNTIL   = 30_000

# 모드 붕괴 감지 & 챔피언 리셋
COLLAPSE_WINDOW   = 200
COLLAPSE_THRESH   = 0.05
RESET_EVERY       = 50_000

class HeuristicBot:
    def decide_next_move(self, engine,
                         temperature=1.0, use_mcts=False):
        board  = engine.board.board
        player = engine.current_player
        opp    = 3 - player
        n      = engine.board_size
        candidates = self._candidates(board, n)
        if not candidates:
            return None
        if player == 1:
            candidates = [(r,c) for r,c in candidates
                          if not Rules.is_forbidden(board, r, c, 1)]
        if not candidates:
            return None
        for length, target in [(5,player),(5,opp),(4,player),(4,opp),(3,player),(3,opp)]:
            move = self._find_threat(board, candidates, target, length, n)
            if move:
                return move
        return self._weighted_random(candidates, n)

    def _candidates(self, board, n):
        occupied = np.argwhere(board != 0)
        if len(occupied) == 0:
            c = n // 2; return [(c,c)]
        seen = set()
        for or_, oc in occupied:
            for dr in range(-2,3):
                for dc in range(-2,3):
                    nr, nc = int(or_)+dr, int(oc)+dc
                    if 0<=nr<n and 0<=nc<n and board[nr,nc]==0 and (nr,nc) not in seen:
                        seen.add((nr,nc))
        return list(seen)

    def _find_threat(self, board, candidates, player, length, n):
        best, best_score = None, -1
        for r,c in candidates:
            board[r,c] = player
            score = self._max_len(board,r,c,player,n)
            board[r,c] = 0
            if score >= length and score > best_score:
                best_score = score; best = (r,c)
        return best

    def _max_len(self, board, r, c, player, n):
        best = 1
        for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
            cnt = 1
            for sign in (1,-1):
                nr,nc = r+dr*sign, c+dc*sign
                while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                    cnt+=1; nr+=dr*sign; nc+=dc*sign
            best = max(best,cnt)
        return best

    def _weighted_random(self, candidates, n):
        if not candidates: return None
        center = n//2
        weights = [1.0/(abs(r-center)+abs(c-center)+1) for r,c in candidates]
        total = sum(weights); weights = [w/total for w in weights]
        idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[idx]

def ensure_dirs():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    try: DRIVE_CKPT.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def save_champion(agent, episode=0):
    ensure_dirs()
    torch.save({'net':agent.net.state_dict(),
                'optimizer':agent.optimizer.state_dict(),
                'episode':episode}, P2_PATH)
    print(f'  [♛] 저장 (ep {episode:,})')

def save_to_drive(episode):
    import shutil
    target = KAGGLE_OUT if KAGGLE_OUT.exists() else DRIVE_CKPT
    try:
        for p in [P2_PATH, LOG_PATH]:
            if p.exists(): shutil.copy(p, target/p.name)
        print(f'  [백업] (ep {episode:,})')
    except Exception as e:
        print(f'  [백업 실패] {e}')

def load_from_drive():
    import shutil
    kaggle_input = Path('/kaggle/input/datasets/hellocarrot/omokgame/files_omokgame/omokGame-test_branch/checkpoints')
    src_dir = kaggle_input if kaggle_input.exists() else DRIVE_CKPT
    for p in [P2_PATH, LOG_PATH]:
        src = src_dir/p.name
        if src.exists():
            ensure_dirs(); shutil.copy(src,p); print(f'  [LOAD] {p.name}')
        else: print(f'  [SKIP] {p.name}')

def load_champion(agent):
    if P2_PATH.exists():
        ckpt = torch.load(P2_PATH, map_location=agent.device)
        agent.net.load_state_dict(ckpt['net'])
        agent.old_net.load_state_dict(ckpt['net'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])
        ep = ckpt.get('episode',0)
        print(f'  [LOAD] ep {ep:,}')
        return ep
    print('  [SKIP] 랜덤 가중치로 시작')
    return 0

def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH,'r',encoding='utf-8') as f: return json.load(f)
    return {
        'meta':{'board_size':BOARD_SIZE,'mode':'Curriculum→Self-Play PPO',
                'created_at':datetime.now().isoformat(timespec='seconds'),'updated_at':''},
        'episodes':[],
        'summary':{'total_episodes':0,'challenger_wins':0,'champion_wins':0,
                   'draws':0,'champion_updates':0},
    }

def append_log(log, record):
    log['episodes'].append(record)
    s = log['summary']; s['total_episodes'] += 1
    w = record.get('challenger_won')
    if w is True: s['challenger_wins'] += 1
    elif w is False: s['champion_wins'] += 1
    else: s['draws'] += 1
    if record.get('champion_updated'): s['champion_updates'] += 1
    log['meta']['updated_at'] = datetime.now().isoformat(timespec='seconds')

def save_log(log):
    ensure_dirs()
    with open(LOG_PATH,'w',encoding='utf-8') as f: json.dump(log,f,ensure_ascii=False,indent=2)

def export_zip(out_dir='.'):
    ensure_dirs()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    zp = Path(out_dir)/f'omok_p2_{ts}.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in [P2_PATH,LOG_PATH]:
            if p.exists(): zf.write(p,arcname=p.name)
    print(f'[EXPORT] → {zp}'); return str(zp)

def _max_len(board, r, c, player, n):
    best = 1
    for dr,dc in [(0,1),(1,0),(1,1),(1,-1)]:
        cnt = 1
        for sign in (1,-1):
            nr,nc = r+dr*sign, c+dc*sign
            while 0<=nr<n and 0<=nc<n and board[nr,nc]==player:
                cnt+=1; nr+=dr*sign; nc+=dc*sign
        best = max(best,cnt)
    return best

def shaped_reward(engine, my_move, opp_move, player):
    board = engine.board.board
    opp   = 3 - player
    n     = engine.board_size

    if engine.is_over:
        if engine.winner == player: return  50.0
        if engine.winner == opp:    return -45.0
        return 0.0

    reward = 0.2
    mr, mc = my_move
    my_len = _max_len(board, mr, mc, player, n)
    if   my_len >= 4: reward += 8.0
    elif my_len == 3: reward += 2.0
    elif my_len == 2: reward += 0.5

    if opp_move is not None:
        or_, oc = opp_move
        opp_len = _max_len(board, or_, oc, opp, n)
        if   opp_len >= 4: reward -= 12.0
        elif opp_len == 3: reward -=  3.0
        elif opp_len == 2: reward -=  0.5

    return float(reward)

def safe_make_move(env, move):
    if move is None: return False
    try: return bool(env.make_move(*move))
    except Exception: return False

def evaluate(challenger, champ_weights, n_games=EVAL_GAMES):
    assert n_games % 2 == 0
    half  = n_games // 2
    champ = challenger.make_champion_agent(champ_weights)
    mem_bak = (list(challenger.memory.states), list(challenger.memory.actions),
               list(challenger.memory.logprobs), list(challenger.memory.values),
               list(challenger.memory.rewards), list(challenger.memory.dones))
    challenger._eval_mode = True

    black_wins = white_wins = 0
    for g in range(n_games):
        print(f'  [EVAL] {g+1}/{n_games}판...', end='\r')  # 진행 로그 추가
        env = Engine(BOARD_SIZE); env.reset()
        ch_color = 1 if g < half else 2
        agents = {ch_color: challenger, 3-ch_color: champ}
        step = invalid = 0
        while not env.is_over and step < MAX_HALF_MOVES:
            cur  = env.current_player
            move = agents[cur].decide_best_move(env)
            if not safe_make_move(env, move):
                invalid += 1
                if invalid >= MAX_INVALID_MOVES: break
                continue
            invalid = 0; step += 1
        if env.winner == ch_color:
            if ch_color == 1: black_wins += 1
            else: white_wins += 1

    print()  # \r 줄 정리
    challenger._eval_mode = False
    (challenger.memory.states, challenger.memory.actions,
     challenger.memory.logprobs, challenger.memory.values,
     challenger.memory.rewards, challenger.memory.dones) = mem_bak
    return {'total':(black_wins+white_wins)/n_games,
            'black':black_wins/half, 'white':white_wins/half}

def print_stats(ep, stats, eval_result, phase):
    bar_len  = 30
    bar      = '█'*int(ep/TOTAL_EPISODES*bar_len) + '░'*(bar_len-int(ep/TOTAL_EPISODES*bar_len))
    chw=stats['challenger_wins']; cpw=stats['champion_wins']
    drw=stats['draws']; tot=max(chw+cpw+drw,1)
    loss=stats.get('loss')
    print(f'\n{"─"*60}')
    print(f'  [{phase}]  ep {ep:>8,} / {TOTAL_EPISODES:,}')
    print(f'  [{bar}] {ep/TOTAL_EPISODES*100:.1f}%')
    if isinstance(loss,float):
        print(f'  스텝:{stats["steps"]:,}  보상:{stats["ep_reward"]:+.2f}  Loss:{loss:.6f}')
    else:
        print(f'  스텝:{stats["steps"]:,}  보상:{stats["ep_reward"]:+.2f}  Loss:—')
    print(f'  {"흑" if stats["ch_color"]==1 else "백"}돌  Ch {chw:,}({chw/tot*100:.1f}%) Opp {cpw:,}({cpw/tot*100:.1f}%) 무 {drw:,}({drw/tot*100:.1f}%)')
    if eval_result:
        tag = '♛ 갱신!' if stats.get('champion_updated') else '─ 유지'
        print(f'  [{tag}] 전체:{eval_result["total"]*100:.1f}% 흑:{eval_result["black"]*100:.1f}% 백:{eval_result["white"]*100:.1f}%')
        print(f'  역대 최고:{stats["best_eval_wr"]*100:.1f}%  갱신:{stats["champion_updates"]}회')
    print(f'{"─"*60}')

def train(resume=False):
    ensure_dirs()
    challenger = PPOAgent(BOARD_SIZE, player=2)
    start_ep = 0
    if resume:
        print('[RESUME] 체크포인트 복사 중...')
        load_from_drive()
        start_ep = load_champion(challenger)

    if not P2_PATH.exists():
        save_champion(challenger, episode=0)
        save_to_drive(0)

    champ_weights = challenger.clone_weights()
    log  = load_log()
    summ = log['summary']
    best_eval_wr  = 0.0
    heuristic_bot = HeuristicBot()

    recent_results: list = []

    stats = {
        'episode':start_ep,'steps':0,
        'challenger_wins':summ['challenger_wins'],
        'champion_wins':summ['champion_wins'],
        'draws':summ['draws'],
        'ep_reward':0.0,'loss':None,'best_eval_wr':0.0,
        'champion_updated':False,'champion_updates':summ['champion_updates'],
        'ch_color':1,
    }

    print(f'\n[TRAIN] Curriculum → Self-Play PPO (4일 T4 최적화)')
    print(f'  장치: {challenger.device}  시작: {start_ep+1:,}  목표: {TOTAL_EPISODES:,}')
    print(f'  Phase1: ~ {HEURISTIC_UNTIL:,} (vs HeuristicBot)')
    print(f'  Phase2: {HEURISTIC_UNTIL+1:,} ~ (Self-Play, 평가 {EVAL_EVERY}ep/{EVAL_GAMES}판)\n')

    last_eval = None

    for ep in range(start_ep+1, TOTAL_EPISODES+1):
        use_heuristic = (ep <= HEURISTIC_UNTIL)
        phase_str     = 'Phase1' if use_heuristic else 'Phase2'

        if ep == HEURISTIC_UNTIL + 1:
            print(f'\n{"="*60}')
            print(f'  [전환] Phase 2 Self-Play 시작 (ep {ep:,})')
            champ_weights = challenger.clone_weights()
            save_champion(challenger, episode=ep)
            print(f'{"="*60}\n')

        ch_color    = 1 if ep%2==1 else 2
        champ_color = 3 - ch_color

        if use_heuristic:
            agents = {ch_color:challenger, champ_color:heuristic_bot}
        else:
            champ_agent = challenger.make_champion_agent(champ_weights)
            agents      = {ch_color:challenger, champ_color:champ_agent}

        env = Engine(BOARD_SIZE); env.reset()
        challenger.memory.clear()
        challenger.reset_episode()
        ep_reward = 0.0; step = 0
        champion_updated  = False
        last_opp_move     = None
        invalid           = 0

        while not env.is_over and step < MAX_HALF_MOVES:
            cur   = env.current_player
            move  = agents[cur].decide_next_move(env)
            if not safe_make_move(env, move):
                invalid += 1
                if invalid >= MAX_INVALID_MOVES: break
                continue
            invalid = 0; step += 1

            if cur == ch_color:
                r = shaped_reward(env, move, last_opp_move, ch_color)
                ep_reward += r
                challenger.store_reward(r, env.is_over)
                last_opp_move = None
            else:
                last_opp_move = move
                challenger.notify_opp_move(move)
                if env.is_over:
                    r = shaped_reward(env, move, None, ch_color)
                    ep_reward += r
                    challenger.store_reward(r, True)

        winner = env.winner
        if winner == ch_color:
            challenger_won = True;  summ['challenger_wins'] += 1
        elif winner == champ_color:
            challenger_won = False; summ['champion_wins']   += 1
        else:
            challenger_won = None;  summ['draws']           += 1

        # 모드 붕괴 감지
        if not use_heuristic:
            recent_results.append(challenger_won is True)
            if len(recent_results) > COLLAPSE_WINDOW:
                recent_results.pop(0)

            win_rate_recent = sum(recent_results) / len(recent_results) if recent_results else 0.5

            if (len(recent_results) >= COLLAPSE_WINDOW
                    and win_rate_recent < COLLAPSE_THRESH):
                print("\n  [붕괴 감지] ep %d 최근 %d판 승률 %.1f%%" % (ep, COLLAPSE_WINDOW, win_rate_recent*100))
                print("  -> 챔피언을 challenger 현재 가중치로 리셋")
                champ_weights = challenger.clone_weights()
                best_eval_wr  = 0.0
                recent_results.clear()
                save_champion(challenger, episode=ep)

            elif (not use_heuristic
                    and ep % RESET_EVERY == 0
                    and win_rate_recent < 0.35):
                print("\n  [주기 리셋] ep %d 최근 승률 %.1f%% -> 챔피언 동기화" % (ep, win_rate_recent*100))
                champ_weights = challenger.clone_weights()
                best_eval_wr  = 0.0
                recent_results.clear()

        if ep % EVAL_EVERY == 0 and not use_heuristic:
            print(f'\n[EVAL] ep {ep:,} — {EVAL_GAMES}판...')
            wr = evaluate(challenger, champ_weights, EVAL_GAMES)
            last_eval = wr
            print(f'  전체:{wr["total"]*100:.1f}% 흑:{wr["black"]*100:.1f}% 백:{wr["white"]*100:.1f}%')
            if wr['total'] >= PROMOTE_WIN_RATE and wr['total'] > best_eval_wr:
                best_eval_wr = wr['total']
                save_champion(challenger, episode=ep)
                save_to_drive(ep)
                champ_weights    = challenger.clone_weights()
                champion_updated = True
                stats['champion_updates'] += 1
                summ['champion_updates']  += 1
                print(f'  [♛] Champion 갱신! ({summ["champion_updates"]}회)')
            else:
                print(f'  [─] 유지 (역대:{best_eval_wr*100:.1f}%)')
            stats['best_eval_wr']     = best_eval_wr
            stats['champion_updated'] = champion_updated
        elif ep % EVAL_EVERY == 0 and use_heuristic:
            save_champion(challenger, episode=ep); save_log(log)

        if ep % SAVE_EVERY       == 0: save_champion(challenger, episode=ep); save_log(log)
        if ep % DRIVE_SAVE_EVERY == 0: save_to_drive(ep); save_log(log)

        record = {
            'episode':ep,'phase':phase_str,'ch_color':ch_color,
            'winner':int(winner) if winner is not None else 0,
            'challenger_won':challenger_won,'steps':step,
            'ep_reward':round(ep_reward,4),'loss':challenger.last_loss,
            'champion_updated':champion_updated,
            'timestamp':datetime.now().isoformat(timespec='seconds'),
        }
        append_log(log, record)
        stats.update({'episode':ep,'steps':step,'ep_reward':ep_reward,
                      'challenger_wins':summ['challenger_wins'],
                      'champion_wins':summ['champion_wins'],
                      'draws':summ['draws'],'loss':challenger.last_loss,
                      'ch_color':ch_color})

        if ep % 200 == 0:
            print_stats(ep, stats, last_eval if ep%EVAL_EVERY==0 else None, phase_str)
            last_eval = None

    save_champion(challenger, episode=ep)
    save_to_drive(ep)
    save_log(log)
    print('\n[DONE] 학습 종료')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.export: export_zip()
    else: train(resume=args.resume)