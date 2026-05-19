import sys
import json
import zipfile
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
import pygame

from ai.engine import Engine
from ai.agent  import PPOAgent
from core.Rules import Rules

# 경로
CKPT_DIR = Path('checkpoints')
P2_PATH  = CKPT_DIR / 'ppo_p2.pt'
LOG_PATH = CKPT_DIR / 'train_log.json'

# 학습 설정
BOARD_SIZE       = 15
TOTAL_EPISODES   = 500000
SAVE_EVERY       = 200
EVAL_EVERY       = 500      # N 에피소드마다 champion 평가
EVAL_GAMES       = 30       # 평가 대결 수
PROMOTE_WIN_RATE = 0.55     # 이 이상이면 champion 갱신

# Pygame UI
CELL     = 30
MARGIN   = 24
BOARD_PX = CELL * (BOARD_SIZE - 1) + MARGIN * 2
PANEL_W  = 340
WIN_W    = BOARD_PX + PANEL_W
WIN_H    = BOARD_PX

C_BG        = ( 18,  18,  28)
C_BOARD     = ( 38,  32,  22)
C_LINE      = ( 90,  70,  40)
C_PANEL     = ( 24,  24,  38)
C_ACCENT    = (  0, 230, 180)
C_WHITE_STN = (240, 240, 250)
C_BLACK_STN = ( 15,  15,  20)
C_RED       = (220,  60,  60)
C_TEXT      = (210, 210, 210)
C_GRAY      = (110, 110, 130)
C_GREEN     = ( 80, 220, 120)
C_YELLOW    = (240, 200,  60)
C_BLUE      = ( 80, 160, 255)

# 파일 유틸
def ensure_dirs():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def save_champion(agent: PPOAgent):
    """현재 모델을 ppo_p2.pt 에 덮어씀 (champion 갱신)."""
    ensure_dirs()
    torch.save({
        'net'      : agent.net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
    }, P2_PATH)
    print(f'  [♛ CHAMPION] ppo_p2.pt 갱신!')


def load_champion(agent: PPOAgent):
    """ppo_p2.pt 를 에이전트에 주입."""
    if P2_PATH.exists():
        ckpt = torch.load(P2_PATH, map_location=agent.device)
        agent.net.load_state_dict(ckpt['net'])
        agent.old_net.load_state_dict(ckpt['net'])
        agent.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'  [LOAD] champion ← {P2_PATH}')
    else:
        print(f'  [SKIP] {P2_PATH} 없음 — 랜덤 가중치로 시작')


def load_log() -> dict:
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
            'challenger_wins' : 0,   # Challenger 가 이긴 횟수 (학습 중)
            'champion_wins'   : 0,   # Champion 이 이긴 횟수
            'draws'           : 0,
            'champion_updates': 0,   # ppo_p2.pt 갱신 횟수
        },
    }


def append_log(log: dict, record: dict):
    log['episodes'].append(record)
    s = log['summary']
    s['total_episodes'] += 1
    w = record.get('challenger_won')
    if w is True : s['challenger_wins'] += 1
    elif w is False: s['champion_wins'] += 1
    else: s['draws'] += 1
    if record.get('champion_updated'):
        s['champion_updates'] += 1
    log['meta']['updated_at'] = datetime.now().isoformat(timespec='seconds')


def save_log(log: dict):
    ensure_dirs()
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def export_zip(out_dir: str = '.') -> str:
    """ppo_p2.pt + train_log.json → ZIP."""
    ensure_dirs()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    zp  = Path(out_dir) / f'omok_p2_{ts}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in [P2_PATH, LOG_PATH]:
            if p.exists():
                zf.write(p, arcname=p.name)
                print(f'  [ZIP] {p.name}')
    print(f'[EXPORT] → {zp}')
    return str(zp)


def import_zip(zip_path: str) -> bool:
    ensure_dirs()
    zp = Path(zip_path)
    if not zp.exists():
        print(f'[ERROR] 파일 없음: {zp}'); return False
    for p in [P2_PATH, LOG_PATH]:
        if p.exists():
            p.rename(p.with_suffix(p.suffix + '.bak'))
    with zipfile.ZipFile(zp, 'r') as zf:
        zf.extractall(CKPT_DIR)
        print(f'  [UNZIP] {zf.namelist()}')
    print(f'[IMPORT] 완료 ← {zp}')
    return True

# 보상 
def shaped_reward(engine: Engine, player: int) -> float:
    board = engine.board.board
    opp   = 3 - player
    if engine.is_over:
        if engine.winner == player : return  50.0
        if engine.winner == opp    : return -50.0
        return 0.0
    r = 0.0
    if Rules.check_patterns(board, player, 4): r += 8.0
    if Rules.check_patterns(board, opp,    4): r -= 15.0
    if Rules.check_patterns(board, player, 3): r += 3.0
    if Rules.check_patterns(board, opp,    3): r -= 5.0
    return float(r)

# Champion 평가
def evaluate(challenger: PPOAgent,
             champ_weights: dict,
             n_games: int = EVAL_GAMES) -> float:
    champ = challenger.make_champion_agent(champ_weights)
    ch_wins = 0

    for g in range(n_games):
        env = Engine(BOARD_SIZE)
        env.reset()

        # 홀수 게임: Challenger=흑(1), Champion=백(2)
        # 짝수 게임: Challenger=백(2), Champion=흑(1)
        if g % 2 == 0:
            ch_color   = 1
            agents     = {1: challenger, 2: champ}
        else:
            ch_color   = 2
            agents     = {1: champ, 2: challenger}

        while not env.is_over:
            cur  = env.current_player
            move = agents[cur].decide_next_move(env)
            if move is None:
                break
            env.make_move(*move)

        if env.winner == ch_color:
            ch_wins += 1

    # 평가 중 쌓인 임시 메모리 비우기
    challenger.memory.clear()
    return ch_wins / n_games

# Pygame 대시보드
class Dashboard:
    MAX_HIST = 200

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('오목 Self-Play PPO')
        self.fL    = pygame.font.SysFont('malgungothic', 22, bold=True)
        self.fM    = pygame.font.SysFont('malgungothic', 15, bold=True)
        self.fS    = pygame.font.SysFont('malgungothic', 12)
        self.clock = pygame.time.Clock()

        # 롤링 그래프 데이터
        self.hist_win_rate = deque(maxlen=self.MAX_HIST)  # 에피소드 승률
        self.hist_eval_wr  = deque(maxlen=self.MAX_HIST)  # 평가 승률

    # 이벤트
    def handle_events(self) -> tuple[bool, bool, bool]:
        """quit, save_now, export_now 반환."""
        quit_f = save_f = export_f = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                quit_f = True
        keys = pygame.key.get_pressed()
        ctrl = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        if keys[pygame.K_q]          : quit_f   = True
        if ctrl and keys[pygame.K_s] : save_f   = True
        if ctrl and keys[pygame.K_e] : export_f = True
        return quit_f, save_f, export_f

    # 전체 렌더링
    def render(self, engine: Engine, stats: dict):
        self.screen.fill(C_BG)
        self._draw_board(engine, stats)
        self._draw_panel(stats)
        pygame.display.flip()
        self.clock.tick(60)

    # 보드
    def _draw_board(self, engine: Engine, stats: dict):
        pygame.draw.rect(self.screen, C_BOARD, (0, 0, BOARD_PX, WIN_H))

        # 격자선
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL
            pygame.draw.line(self.screen, C_LINE,
                (x, MARGIN), (x, MARGIN+(BOARD_SIZE-1)*CELL), 1)
            pygame.draw.line(self.screen, C_LINE,
                (MARGIN, x), (MARGIN+(BOARD_SIZE-1)*CELL, x), 1)

        # 화점
        for p in [3, 7, 11]:
            for q in [3, 7, 11]:
                pygame.draw.circle(self.screen, C_LINE,
                    (MARGIN+q*CELL, MARGIN+p*CELL), 3)

        # 돌
        board = engine.numpy_board
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cx, cy = MARGIN+c*CELL, MARGIN+r*CELL
                v = board[r, c]
                if v == 1:
                    pygame.draw.circle(
                        self.screen, C_BLACK_STN, (cx, cy), 12)
                elif v == 2:
                    pygame.draw.circle(
                        self.screen, C_WHITE_STN, (cx, cy), 12)
                    pygame.draw.circle(
                        self.screen, C_LINE, (cx, cy), 12, 1)

        # 현재 에피소드에서 Challenger 색 표시
        ch_color = stats.get('ch_color', '?')
        label    = self.fS.render(
            f'Challenger = {"흑돌" if ch_color == 1 else "백돌"}',
            True, C_ACCENT)
        self.screen.blit(label, (MARGIN, WIN_H - 20))

    # 패널
    def _draw_panel(self, s: dict):
        px = BOARD_PX
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, PANEL_W, WIN_H))

        y = 14
        def txt(text, color=C_TEXT, font=None, cx=False):
            nonlocal y
            f   = font or self.fS
            sur = f.render(text, True, color)
            bx  = px + (PANEL_W - sur.get_width())//2 if cx else px + 14
            self.screen.blit(sur, (bx, y))
            y  += sur.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, C_GRAY,
                (px+10, y), (px+PANEL_W-10, y), 1)
            y += 7

        # 타이틀
        txt('Self-Play PPO', C_ACCENT, self.fL, cx=True)
        txt('Challenger  vs  Champion', C_GRAY, cx=True)
        sep()

        # 에피소드
        ep = s.get('episode', 0)
        txt(f'에피소드  {ep:,} / {TOTAL_EPISODES:,}', C_TEXT, self.fM)
        txt(f'스텝 수   {s.get("steps", 0):,}')
        sep()

        # 학습 중 승패 (에피소드 누적)
        chw = s.get('challenger_wins', 0)
        cpw = s.get('champion_wins',   0)
        drw = s.get('draws', 0)
        tot = max(chw + cpw + drw, 1)
        txt('학습 중 승패', C_YELLOW, self.fM)
        txt(f'Challenger 승  {chw:,}  ({chw/tot*100:.1f}%)', C_GREEN)
        txt(f'Champion   승  {cpw:,}  ({cpw/tot*100:.1f}%)', C_BLUE)
        txt(f'무승부         {drw:,}  ({drw/tot*100:.1f}%)', C_GRAY)
        sep()

        # Champion 평가 결과
        wr = s.get('eval_win_rate', None)
        if wr is not None:
            color = C_GREEN if wr >= PROMOTE_WIN_RATE else C_RED
            tag   = '♛ champion 갱신!' if s.get('champion_updated') else '유지'
            txt(f'평가 승률  {wr*100:.1f}%  ({tag})', color, self.fM)
        else:
            pct = (ep % EVAL_EVERY) / EVAL_EVERY * 100
            txt(f'다음 평가까지  {EVAL_EVERY - ep % EVAL_EVERY}ep  ({pct:.0f}%)',
                C_GRAY)
        txt(f'총 champion 갱신  {s.get("champion_updates", 0)}회', C_BLUE)
        sep()

        # 보상 / Loss
        txt('최근 에피소드', C_YELLOW, self.fM)
        txt(f'보상    {s.get("ep_reward", 0.0):+.2f}')
        loss = s.get('loss', None)
        txt(f'Loss    {loss:.6f}' if isinstance(loss, float) else 'Loss    —')
        sep()

        # 평가 승률 그래프
        wr_v = s.get('eval_win_rate')
        if wr_v is not None:
            self.hist_eval_wr.append(wr_v)
        self._mini_graph(px+10, y, PANEL_W-20, 54,
                         self.hist_eval_wr, C_GREEN, '평가 승률 추이')
        y += 60

        # 기준선 표시 (PROMOTE_WIN_RATE)
        pct_txt = self.fS.render(
            f'갱신 기준: {PROMOTE_WIN_RATE*100:.0f}%', True, C_GRAY)
        self.screen.blit(pct_txt, (px+14, y)); y += 18
        sep()

        # 단축키
        txt('[Ctrl+S] 즉시 저장', C_GRAY)
        txt('[Ctrl+E] ZIP 내보내기', C_GRAY)
        txt('[Q] 저장 후 종료', C_GRAY)

    # 미니 그래프
    def _mini_graph(self, x, y, w, h, data, color, label):
        pygame.draw.rect(self.screen, (28, 28, 44), (x, y, w, h))
        pygame.draw.rect(self.screen, C_GRAY,       (x, y, w, h), 1)

        # 기준선 (PROMOTE_WIN_RATE)
        base_y = y + h - 2 - int(PROMOTE_WIN_RATE * (h - 14))
        pygame.draw.line(self.screen, C_YELLOW,
                         (x+1, base_y), (x+w-1, base_y), 1)

        lbl = self.fS.render(label, True, C_GRAY)
        self.screen.blit(lbl, (x+4, y+2))

        pts = list(data)
        if len(pts) < 2:
            return
        mn, mx = 0.0, 1.0          # 승률은 항상 0~1 고정 스케일
        rng    = mx - mn
        xs  = [x + int(i/(len(pts)-1)*(w-2))+1 for i in range(len(pts))]
        ys  = [y + h - 2 - int((v-mn)/rng*(h-14))  for v in pts]
        pygame.draw.lines(self.screen, color, False, list(zip(xs, ys)), 1)

# 메인 학습 루프
def train(resume: bool = False):
    ensure_dirs()

    # ── 에이전트 생성
    challenger = PPOAgent(BOARD_SIZE, player=2)

    if resume:
        print('[RESUME] 체크포인트 불러오는 중...')
        load_champion(challenger)

    # 첫 실행이면 랜덤 가중치를 champion 으로 저장
    if not P2_PATH.exists():
        save_champion(challenger)

    # champion 가중치 스냅샷 (메모리에만 보관 — 파일 갱신 없이 비교용)
    champ_weights = challenger.clone_weights()

    # 로그
    log  = load_log()
    summ = log['summary']

    # 대시보드
    dash = Dashboard()
    env  = Engine(BOARD_SIZE)

    # 통계
    stats: dict = {
        'episode'          : summ['total_episodes'],
        'steps'            : 0,
        'challenger_wins'  : summ['challenger_wins'],
        'champion_wins'    : summ['champion_wins'],
        'draws'            : summ['draws'],
        'ep_reward'        : 0.0,
        'loss'             : None,
        'eval_win_rate'    : None,
        'champion_updated' : False,
        'champion_updates' : summ['champion_updates'],
        'ch_color'         : 1,
    }

    print(f'[TRAIN] Self-Play PPO 시작 — {TOTAL_EPISODES:,} 에피소드')
    print(f'        장치: {challenger.device}')
    print(f'        방식: Challenger vs Champion (Self-Play Curriculum)')

    for ep in range(1, TOTAL_EPISODES + 1):

        # 에피소드 색 배정 (번갈아 가며 편향 방지)
        ch_color   = 1 if ep % 2 == 1 else 2   # Challenger 색
        champ_color = 3 - ch_color              # Champion 색

        # champion 추론 에이전트 (가중치 스냅샷 사용)
        champ_agent = challenger.make_champion_agent(champ_weights)
        agents      = {ch_color: challenger, champ_color: champ_agent}

        # 에피소드 진행
        env.reset()
        challenger.memory.clear()
        ep_reward        = 0.0
        step             = 0
        champion_updated = False

        while not env.is_over:
            cur   = env.current_player
            agent = agents[cur]
            move  = agent.decide_next_move(env)
            if move is None:
                break
            env.make_move(*move)
            step += 1

            # Challenger 관점 보상만 계산·저장
            if cur == ch_color:
                r = shaped_reward(env, ch_color)
                ep_reward += r
                challenger.store_reward(r, env.is_over)

        # 학습 중 승패 집계
        winner = env.winner
        if winner == ch_color:
            challenger_won = True
            summ['challenger_wins'] += 1
        elif winner == champ_color:
            challenger_won = False
            summ['champion_wins'] += 1
        else:
            challenger_won = None   # 무승부
            summ['draws'] += 1

        # champion 평가
        if ep % EVAL_EVERY == 0:
            print(f'\n[EVAL]  에피소드 {ep:,} — champion 평가 {EVAL_GAMES}판...')
            wr = evaluate(challenger, champ_weights, EVAL_GAMES)
            print(f'        Challenger 승률: {wr*100:.1f}%  '
                  f'(기준 {PROMOTE_WIN_RATE*100:.0f}%)')

            if wr >= PROMOTE_WIN_RATE:
                save_champion(challenger)
                champ_weights    = challenger.clone_weights()
                champion_updated = True
                stats['champion_updates'] += 1
                summ['champion_updates']  += 1
                print(f'        [♛] Champion 갱신 완료 '
                      f'(총 {summ["champion_updates"]}회)')
            else:
                print(f'        [─] Champion 유지')

            stats['eval_win_rate']    = wr
            stats['champion_updated'] = champion_updated

        # 주기 저장
        if ep % SAVE_EVERY == 0:
            save_log(log)

        # 로그 추가
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

        # 통계 갱신
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

        # 대시보드 렌더링 + 단축키
        dash.render(env, stats)
        quit_f, save_f, export_f = dash.handle_events()

        if save_f:
            save_champion(challenger)
            save_log(log)
            print(f'[SAVE] 수동 저장 (에피소드 {ep:,})')
        if export_f:
            export_zip()
        if quit_f:
            print('\n[QUIT] 저장 후 종료')
            break

    # 종료 후 최종 저장
    save_champion(challenger)
    save_log(log)
    print('[DONE] 학습 종료 — 최종 저장 완료')
    pygame.quit()

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='오목 Self-Play PPO 학습')
    parser.add_argument('--resume',     action='store_true',
                        help='기존 ppo_p2.pt 이어서 학습')
    parser.add_argument('--export',     action='store_true',
                        help='ZIP 내보내기만 실행')
    parser.add_argument('--import-zip', metavar='ZIP_PATH',
                        help='ZIP 불러오기만 실행')
    args = parser.parse_args()

    if args.export:
        export_zip()
    elif getattr(args, 'import_zip', None):
        import_zip(args.import_zip)
    else:
        train(resume=args.resume)