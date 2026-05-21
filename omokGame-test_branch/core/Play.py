import sys
from pathlib import Path

import pygame

from core.Board import Board
from core.Rules import Rules
from ai.engine  import Engine
from ai.agent   import PPOAgent

# 학습된 AI 모델 경로
CKPT_PATH = Path('checkpoints/ppo_p2.pt')

# 색상
BG_COLOR     = (255, 250, 233)
BOARD_COLOR  = (220, 180, 104)
LINE_COLOR   = (139, 105,  20)
BLACK_COLOR  = (  0,   0,   0)
WHITE_COLOR  = (255, 255, 255)
RED_COLOR    = (220,  50,  50)
PANEL_COLOR  = (  0,   0,   0)
PANEL_FG     = (255, 250, 233)
GRAY_COLOR   = (170, 170, 170)
GREEN_COLOR  = ( 80, 200, 120)

# 화면 및 보드 설정
CELL_SIZE    = 38
BOARD_SIZE   = 15
MARGIN       = 30
CANVAS_SIZE  = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
PANEL_W      = 240
WIN_W        = CANVAS_SIZE + PANEL_W
WIN_H        = CANVAS_SIZE

# 버튼 컴포넌트 클래스
class Button:
    def __init__(self, rect, text, font, bg, fg, border=False):
        self.rect   = pygame.Rect(rect)
        self.text   = text
        self.font   = font
        self.bg     = bg
        self.fg     = fg
        self.border = border

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg, self.rect, border_radius=4)
        if self.border:
            pygame.draw.rect(surface, self.fg, self.rect, 1, border_radius=4)
        label = self.font.render(self.text, True, self.fg)
        lx = self.rect.centerx - label.get_width()  // 2
        ly = self.rect.centery - label.get_height() // 2
        surface.blit(label, (lx, ly))

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))

# 메인 게임 클래스
class Play:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('오목 게임')

        self.font_large  = pygame.font.SysFont('malgungothic', 36, bold=True)
        self.font_medium = pygame.font.SysFont('malgungothic', 20, bold=True)
        self.font_small  = pygame.font.SysFont('malgungothic', 14)

        self.scene        = 'start'
        self.screen_state = 'start'
        self.mode         = None
        self.human_player = 1
        self.engine       = None
        self.agent        = None
        self.last_move    = None
        self.ai_pending   = False
        self.ai_timer     = 0
        self.clock        = pygame.time.Clock()

        # 체크포인트 존재 여부 미리 확인
        self.ckpt_exists  = CKPT_PATH.exists()

        self._build_start_buttons()

    # 시작화면 버튼
    def _build_start_buttons(self):
        cx = WIN_W // 2
        self.btn_human = Button(
            (cx - 120, 260, 240, 56),
            '인간  vs  인간',
            self.font_medium,
            BLACK_COLOR, BG_COLOR,
        )
        self.btn_ai = Button(
            (cx - 120, 332, 240, 56),
            '인간  vs  AI',
            self.font_medium,
            WHITE_COLOR, BLACK_COLOR,
            border=True
        )

    # 처음으로 버튼
    def _build_game_buttons(self):
        bx = CANVAS_SIZE + (PANEL_W - 160) // 2
        self.btn_home = Button(
            (bx, WIN_H - 80, 160, 44),
            '처음으로',
            self.font_small,
            BG_COLOR, BLACK_COLOR,
        )

    # 메인 루프
    def run(self):
        while True:
            dt = self.clock.tick(60)
            self._handle_events()
            self._update(dt)
            self._draw()
            pygame.display.flip()

    # 이벤트 처리
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if self.screen_state == 'start':
                self._handle_start(event)
            else:
                self._handle_game(event)

    # 매 프레임 상태 업데이트
    def _update(self, dt):
        if self.ai_pending:
            if pygame.time.get_ticks() - self.ai_timer >= 300:
                self.ai_pending = False
                self._ai_move()

    # 시작 화면 이벤트
    def _handle_start(self, event):
        if self.btn_human.is_clicked(event):
            self._start_human_game()
        elif self.btn_ai.is_clicked(event):
            self._start_ai_game()

    # 게임 화면 이벤트
    def _handle_game(self, event):
        if self.btn_home.is_clicked(event):
            self.screen_state = 'start'
            self._build_start_buttons()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.engine.is_over or self.ai_pending:
                return
            if (self.mode == 'ai'
                    and self.engine.current_player != self.human_player):
                return
            mx, my = event.pos
            j = round((mx - MARGIN) / CELL_SIZE)
            i = round((my - MARGIN) / CELL_SIZE)

            if not (0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE):
                return
            if self.engine.make_move(i, j):
                self.last_move = (i, j)
                self._check_game_over()
                if self.mode == 'ai' and not self.engine.is_over:
                    self.ai_pending = True
                    self.ai_timer   = pygame.time.get_ticks()

    # AI 착수 — 대국 중 학습 없음 (train.py 에서만 학습)
    def _ai_move(self):
        if self.agent is None or self.engine.is_over:
            return
        if self.engine.current_player != 2:  # AI = 백돌(2) 고정
            return

        move = self.agent.decide_next_move(self.engine)
        if move:
            self.engine.make_move(*move)
            self.last_move = move
            self._check_game_over()

    # 게임 종료 처리
    def _check_game_over(self):
        if not self.engine.is_over:
            return

        if self.engine.winner == 1:
            msg = '흑돌(인간) 승리!'
        elif self.engine.winner == 2:
            msg = '백돌(AI) 승리!'
        else:
            msg = '무승부!'

        self._draw_game()
        pygame.display.flip()
        self._show_dialog(msg)
        self.screen_state = 'start'
        self._build_start_buttons()

    # 결과 팝업
    def _show_dialog(self, msg):
        box_w, box_h = 340, 180
        bx = (WIN_W - box_w) // 2
        by = (WIN_H - box_h) // 2

        pygame.draw.rect(self.screen, BG_COLOR,
                         (bx, by, box_w, box_h), border_radius=8)
        pygame.draw.rect(self.screen, LINE_COLOR,
                         (bx, by, box_w, box_h), 2, border_radius=8)

        title = self.font_medium.render('게임 종료', True, BLACK_COLOR)
        self.screen.blit(title,
            (bx + box_w // 2 - title.get_width() // 2, by + 30))

        body = self.font_medium.render(msg, True, BLACK_COLOR)
        self.screen.blit(body,
            (bx + box_w // 2 - body.get_width() // 2, by + 80))

        ok_btn = Button(
            (bx + box_w // 2 - 60, by + 128, 120, 36),
            '확인', self.font_small, BLACK_COLOR, BG_COLOR)
        ok_btn.draw(self.screen)
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if (event.type == pygame.MOUSEBUTTONDOWN
                        and ok_btn.is_clicked(event)):
                    return
                if event.type == pygame.KEYDOWN:
                    return

    # 인간 vs 인간 게임 시작
    def _start_human_game(self):
        self.mode         = 'human'
        self.human_player = 1
        self.engine       = Engine(BOARD_SIZE)
        self.agent        = None
        self.last_move    = None
        self.ai_pending   = False
        self.screen_state = 'game'
        self._build_game_buttons()

    # 인간 vs AI 게임 시작 — 학습된 가중치 로드
    def _start_ai_game(self):
        self.mode         = 'ai'
        self.human_player = 1           # 인간 = 흑돌(1) 고정
        self.engine       = Engine(BOARD_SIZE)
        self.last_move    = None
        self.ai_pending   = False

        # AI 에이전트 생성 및 가중치 로드
        self.agent = PPOAgent(BOARD_SIZE, player=2)  # AI = 백돌(2) 고정
        if CKPT_PATH.exists():
            self.agent.load(str(CKPT_PATH))
            print(f'[AI] 체크포인트 로드 완료 ← {CKPT_PATH}')
        else:
            print(f'[AI] 체크포인트 없음 — 랜덤 가중치로 시작')

        self.screen_state = 'game'
        self._build_game_buttons()

    # 그리기
    def _draw(self):
        if self.screen_state == 'start':
            self._draw_start()
        else:
            self._draw_game()

    # 시작 화면
    def _draw_start(self):
        self.screen.fill(BG_COLOR)

        title = self.font_large.render('오목 게임', True, BLACK_COLOR)
        self.screen.blit(title,
            (WIN_W // 2 - title.get_width() // 2, 180))

        self.btn_human.draw(self.screen)
        self.btn_ai.draw(self.screen)

        # AI 모델 로드 상태 표시
        if self.ckpt_exists:
            status = self.font_small.render('✓ AI 모델 로드됨', True, GREEN_COLOR)
        else:
            status = self.font_small.render('⚠ AI 모델 없음 (랜덤)', True, RED_COLOR)
        self.screen.blit(status,
            (WIN_W // 2 - status.get_width() // 2, 400))

    # 게임 화면
    def _draw_game(self):
        self.screen.fill(BG_COLOR)
        self._draw_board_area()
        self._draw_panel()

    # 바둑판 영역
    def _draw_board_area(self):
        pygame.draw.rect(self.screen, BOARD_COLOR,
                         (0, 0, CANVAS_SIZE, WIN_H))

        for i in range(BOARD_SIZE):
            x0 = MARGIN + i * CELL_SIZE
            pygame.draw.line(self.screen, LINE_COLOR,
                (x0, MARGIN), (x0, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE), 1)
            pygame.draw.line(self.screen, LINE_COLOR,
                (MARGIN, x0), (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, x0), 1)

        for sr in [3, 7, 11]:
            for sc in [3, 7, 11]:
                cx = MARGIN + sc * CELL_SIZE
                cy = MARGIN + sr * CELL_SIZE
                pygame.draw.circle(self.screen, LINE_COLOR, (cx, cy), 4)

        # 금수 위치에 X 표시 (흑돌 차례일 때)
        if (self.engine.current_player == 1
                and not self.engine.is_over):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (self.engine.board.board[i][j] == 0
                            and Rules.is_forbidden(
                                self.engine.board.board, i, j, 1)):
                        cx = MARGIN + j * CELL_SIZE
                        cy = MARGIN + i * CELL_SIZE
                        pygame.draw.line(self.screen, RED_COLOR,
                            (cx - 7, cy - 7), (cx + 7, cy + 7), 2)
                        pygame.draw.line(self.screen, RED_COLOR,
                            (cx + 7, cy - 7), (cx - 7, cy + 7), 2)

        # 돌 그리기
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cx = MARGIN + j * CELL_SIZE
                cy = MARGIN + i * CELL_SIZE
                v  = self.engine.board.board[i][j]
                if v == 1:
                    pygame.draw.circle(self.screen, BLACK_COLOR, (cx, cy), 16)
                elif v == 2:
                    pygame.draw.circle(self.screen, WHITE_COLOR, (cx, cy), 16)
                    pygame.draw.circle(self.screen, LINE_COLOR,  (cx, cy), 16, 1)

        # 마지막 착수 위치에 빨간 점 표시
        if self.last_move:
            li, lj = self.last_move
            cx = MARGIN + lj * CELL_SIZE
            cy = MARGIN + li * CELL_SIZE
            pygame.draw.circle(self.screen, RED_COLOR, (cx, cy), 5)

    # 우측 패널
    def _draw_panel(self):
        px = CANVAS_SIZE
        pygame.draw.rect(self.screen, PANEL_COLOR, (px, 0, PANEL_W, WIN_H))

        title = self.font_medium.render('오목 게임', True, PANEL_FG)
        self.screen.blit(title,
            (px + PANEL_W // 2 - title.get_width() // 2, 24))

        pygame.draw.line(self.screen, GRAY_COLOR,
            (px + 20, 64), (px + PANEL_W - 20, 64), 1)

        # 흑/백 역할 표시
        role_b = self.font_small.render('● 흑돌 — 인간', True, GRAY_COLOR)
        role_w = self.font_small.render('○ 백돌 — AI',   True, GRAY_COLOR)
        self.screen.blit(role_b, (px + 20, 76))
        self.screen.blit(role_w, (px + 20, 96))

        pygame.draw.line(self.screen, GRAY_COLOR,
            (px + 20, 120), (px + PANEL_W - 20, 120), 1)

        sub = self.font_small.render('현재 차례', True, GRAY_COLOR)
        self.screen.blit(sub,
            (px + PANEL_W // 2 - sub.get_width() // 2, 130))

        if self.engine.is_over:
            if self.engine.winner == 1:
                turn_text = '흑돌 승리!'
            elif self.engine.winner == 2:
                turn_text = 'AI 승리!'
            else:
                turn_text = '무승부!'
        else:
            turn_text = ('흑돌 (인간)'
                         if self.engine.current_player == 1
                         else '백돌 (AI)')

        turn = self.font_medium.render(turn_text, True, WHITE_COLOR)
        self.screen.blit(turn,
            (px + PANEL_W // 2 - turn.get_width() // 2, 155))

        pygame.draw.line(self.screen, GRAY_COLOR,
            (px + 20, 200), (px + PANEL_W - 20, 200), 1)

        mode_str = ('인간 vs 인간' if self.mode == 'human' else '인간 vs AI')
        mode_lbl = self.font_small.render(mode_str, True, GRAY_COLOR)
        self.screen.blit(mode_lbl,
            (px + PANEL_W // 2 - mode_lbl.get_width() // 2, 215))

        if self.ai_pending:
            wait = self.font_small.render('AI 생각 중...', True, GRAY_COLOR)
            self.screen.blit(wait,
                (px + PANEL_W // 2 - wait.get_width() // 2, 240))

        # 체크포인트 로드 상태
        if self.mode == 'ai':
            ckpt_color = GREEN_COLOR if self.ckpt_exists else RED_COLOR
            ckpt_text  = '모델 로드됨' if self.ckpt_exists else '랜덤 가중치'
            ckpt_lbl = self.font_small.render(ckpt_text, True, ckpt_color)
            self.screen.blit(ckpt_lbl,
                (px + PANEL_W // 2 - ckpt_lbl.get_width() // 2, 265))

        self.btn_home.draw(self.screen)


if __name__ == '__main__':
    Play().run()