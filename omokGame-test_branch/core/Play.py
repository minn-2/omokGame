import sys
import pygame

from core.Board import Board
from core.Rules import Rules
from ai.engine  import Engine
from ai.agent   import PPOAgent

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

# 화면 및 보드 설정
CELL_SIZE    = 38 # 격자 한 칸
BOARD_SIZE   = 15 
MARGIN       = 30 # 바둑판 외곽 여백
CANVAS_SIZE  = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2 # 바둑판 영역 전체 픽셀 크기
PANEL_W      = 240 # 우측 패널 너비
WIN_W        = CANVAS_SIZE + PANEL_W # 창 전체 너비
WIN_H        = CANVAS_SIZE # 창 전체 높이

# 버튼 컴포넌트 클래스
class Button:
    def __init__(self, rect, text, font, bg, fg, border=False):
        self.rect   = pygame.Rect(rect) # 버튼 영역(x, y, 너비, 높이)
        self.text   = text # 버튼 텍스트
        self.font   = font 
        self.bg     = bg
        self.fg     = fg
        self.border = border

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg, self.rect, border_radius=4)

        if self.border:
            pygame.draw.rect(surface, self.fg, self.rect, 1, border_radius=4)

        label = self.font.render(self.text, True, self.fg) # 텍스트 렌더링
        lx = self.rect.centerx - label.get_width()  // 2 # 텍스트 중앙 배치
        ly = self.rect.centery - label.get_height() // 2
        surface.blit(label, (lx, ly))
    # 마우스 왼쪽 버튼 클릭 여부 반환
    def is_clicked(self, event): 
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))

# 메인 게임 클래스
class Play:
    def __init__(self):
        pygame.init() # 초기화
        self.screen = pygame.display.set_mode((WIN_W, WIN_H)) # 게임 창 생성
        pygame.display.set_caption('오목 게임') # 창 제목 설정

        self.font_large  = pygame.font.SysFont('malgungothic', 36, bold=True)
        self.font_medium = pygame.font.SysFont('malgungothic', 20, bold=True)
        self.font_small  = pygame.font.SysFont('malgungothic', 14)
        # 게임 상태 변수
        self.scene        = 'start' # 현재 화면 상태
        self.screen_state = 'start' # 화면 전환 제어
        self.mode         = None # 게임 모드
        self.human_player = 1 # 인간 플레이어 번호
        self.engine       = None # 게임 엔진(보드 상태, 착수, 승패 판정)
        self.agent        = None # PPO 에이전트
        self.last_move    = None # 마지막 착수 위치
        self.ai_pending   = False # AI 착수 딜레이
        self.ai_timer     = 0 # AI 착수 대기 시간
        self.clock        = pygame.time.Clock() # 프레임 조절용 시간
 
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
    # 메인 루프 : 이벤트처리, 상태 업데이트, 화면 렌더링 반복
    def run(self):
        while True:
            dt = self.clock.tick(60) # 60프레임 제한
            self._handle_events() 
            self._update(dt)
            self._draw()
            pygame.display.flip() # 화면 갱신
    # 이벤트 처리
    def _handle_events(self):
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: # 창 닫기 버튼 클릭 시 종료
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
            # ai 모드에서 ai 차례면 입력 무시
            if (self.mode == 'ai'
                    and self.engine.current_player != self.human_player):
                return
            # 픽셀 좌표 보드 격자 좌표 변환 (반올림)
            mx, my = event.pos
            j = round((mx - MARGIN) / CELL_SIZE)
            i = round((my - MARGIN) / CELL_SIZE)

            if not (0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE):
                return
            # 착수 시도
            if self.engine.make_move(i, j):
                self.last_move = (i, j)
                self._check_game_over()
                # ai 모드이고 게임이 진행중이면 ai착수 예약
                if self.mode == 'ai' and not self.engine.is_over:
                    self.ai_pending = True
                    self.ai_timer   = pygame.time.get_ticks()
    # au 착수
    def _ai_move(self):
        if self.agent is None or self.engine.is_over: # 에이전트 없거나 게임 종료일 때
            return
        if self.engine.current_player != self.agent.player: # AI 차례가 아닐 때
            return

        move = self.agent.decide_next_move(self.engine)
        if move:
            self.engine.make_move(*move)
            self.last_move = move
            reward = self.agent.calculate_reward(self.engine)
            self.agent.store_reward(reward)
            self._check_game_over()
    # 게임 종료 처리
    def _check_game_over(self):
        if not self.engine.is_over:
            return

        if self.engine.winner == 1:
            msg = '흑돌 승리!'
        elif self.engine.winner == 2:
            msg = '백돌 승리!'
        else:
            msg = '무승부!'

        # 마지막 돌이 화면에 그려진 후 팝업 표시
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
        # 팝업 배경
        pygame.draw.rect(
            self.screen, BG_COLOR,
            (bx, by, box_w, box_h), border_radius=8)
        # 팝업 테두리
        pygame.draw.rect(
            self.screen, LINE_COLOR,
            (bx, by, box_w, box_h), 2, border_radius=8)

        title = self.font_medium.render('게임 종료', True, BLACK_COLOR)
        self.screen.blit(
            title,
            (bx + box_w // 2 - title.get_width() // 2, by + 30))

        body = self.font_medium.render(msg, True, BLACK_COLOR)
        self.screen.blit(
            body,
            (bx + box_w // 2 - body.get_width() // 2, by + 80))

        ok_btn = Button(
            (bx + box_w // 2 - 60, by + 128, 120, 36),
            '확인',
            self.font_small,
            BLACK_COLOR, BG_COLOR,
        )
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
    # 게임 시작
    def _start_human_game(self):
        self.mode         = 'human'
        self.human_player = 1
        self.engine       = Engine(BOARD_SIZE)
        self.agent        = None
        self.last_move    = None
        self.ai_pending   = False
        self.screen_state = 'game'
        self._build_game_buttons()

    def _start_ai_game(self):
        self.mode         = 'ai'
        self.human_player = 1
        self.engine       = Engine(BOARD_SIZE)
        self.last_move    = None
        self.ai_pending   = False
        self.agent        = PPOAgent(BOARD_SIZE, player=2)
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
        self.screen.blit(
            title,
            (WIN_W // 2 - title.get_width() // 2, 180))

        self.btn_human.draw(self.screen)
        self.btn_ai.draw(self.screen)
    # 게임 화면(바둑판, 우측 패널)
    def _draw_game(self):
        self.screen.fill(BG_COLOR)
        self._draw_board_area()
        self._draw_panel()
    # 바둑판 화면(배경, 격자선, 화점, 금수 표시, 돌, 마지막 착수 표시) 
    def _draw_board_area(self):
        pygame.draw.rect(
            self.screen, BOARD_COLOR,
            (0, 0, CANVAS_SIZE, WIN_H))
        # 격자선 그리기
        for i in range(BOARD_SIZE):
            x0 = MARGIN + i * CELL_SIZE
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (x0, MARGIN),
                (x0, MARGIN + (BOARD_SIZE - 1) * CELL_SIZE), 1)
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN, x0),
                (MARGIN + (BOARD_SIZE - 1) * CELL_SIZE, x0), 1)
        # 화점 그리기
        for sr in [3, 7, 11]:
            for sc in [3, 7, 11]:
                cx = MARGIN + sc * CELL_SIZE
                cy = MARGIN + sr * CELL_SIZE
                pygame.draw.circle(
                    self.screen, LINE_COLOR, (cx, cy), 4)
        # 금수 위치에 X표시
        if (self.engine.current_player == 1
                and not self.engine.is_over):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (self.engine.board.board[i][j] == 0
                            and Rules.is_forbidden(
                                self.engine.board.board, i, j, 1)):
                        cx = MARGIN + j * CELL_SIZE
                        cy = MARGIN + i * CELL_SIZE
                        pygame.draw.line(
                            self.screen, RED_COLOR,
                            (cx - 7, cy - 7), (cx + 7, cy + 7), 2)
                        pygame.draw.line(
                            self.screen, RED_COLOR,
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
        # 마지막 착수 위치에 빨간 점 표시
        if self.last_move:
            li, lj = self.last_move
            cx = MARGIN + lj * CELL_SIZE
            cy = MARGIN + li * CELL_SIZE
            pygame.draw.circle(self.screen, RED_COLOR, (cx, cy), 5)
    # 우측 패널
    def _draw_panel(self):
        px = CANVAS_SIZE
        pygame.draw.rect(
            self.screen, PANEL_COLOR,
            (px, 0, PANEL_W, WIN_H))

        # 타이틀
        title = self.font_medium.render('오목 게임', True, PANEL_FG)
        self.screen.blit(
            title,
            (px + PANEL_W // 2 - title.get_width() // 2, 24))
        # 구분선
        pygame.draw.line(
            self.screen, GRAY_COLOR,
            (px + 20, 64), (px + PANEL_W - 20, 64), 1)
        # 현재 차례 레이블
        sub = self.font_small.render('현재 차례', True, GRAY_COLOR)
        self.screen.blit(
            sub,
            (px + PANEL_W // 2 - sub.get_width() // 2, 100))
        # 현재 차례 또는 게임 결과
        if self.engine.is_over:
            if self.engine.winner == 1:
                turn_text = '흑돌 승리!'
            elif self.engine.winner == 2:
                turn_text = '백돌 승리!'
            else:
                turn_text = '무승부!'
        else:
            turn_text = '흑돌' if self.engine.current_player == 1 else '백돌'

        turn = self.font_medium.render(turn_text, True, WHITE_COLOR)
        self.screen.blit(
            turn,
            (px + PANEL_W // 2 - turn.get_width() // 2, 128))

        pygame.draw.line(
            self.screen, GRAY_COLOR,
            (px + 20, 180), (px + PANEL_W - 20, 180), 1)

        mode_str = ('인간 vs 인간' if self.mode == 'human' else '인간 vs AI')
        mode_lbl = self.font_small.render(mode_str, True, GRAY_COLOR)
        self.screen.blit(
            mode_lbl,
            (px + PANEL_W // 2 - mode_lbl.get_width() // 2, 200))
        # AI 대기 중 안내 텍스트
        if self.ai_pending:
            wait = self.font_small.render('AI 생각 중...', True, GRAY_COLOR)
            self.screen.blit(
                wait,
                (px + PANEL_W // 2 - wait.get_width() // 2, 230))

        self.btn_home.draw(self.screen)


if __name__ == '__main__':
        Play().run()
        