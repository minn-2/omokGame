import pygame
import sys

from core.Board import Board
from core.Rules import Rules
from ai.engine  import Engine
from ai.agent   import PPOAgent

# 색상
BG_COLOR    = (255, 250, 233)  
BOARD_COLOR = (220, 180, 104)  
LINE_COLOR  = (139, 105,  20) 
BLACK_COLOR = (  0,   0,   0)  
WHITE_COLOR = (255, 255, 255)  
PANEL_COLOR = (  0,   0,   0)  
TEXT_COLOR  = (255, 250, 233)  
GRAY_COLOR  = (170, 170, 170) 
RED_COLOR   = (200,  50,  50)  


# 화면 및 보드
CELL_SIZE   = 38 
BOARD_SIZE  = 15
MARGIN      = 30 
PANEL_W     = 220
CANVAS_SIZE = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
WIN_W       = CANVAS_SIZE + PANEL_W  # 전체 창 너비
WIN_H       = CANVAS_SIZE # 전체 창 높이


class Play:
    def __init__(self):
        pygame.init() # pygame 초기화
        self.screen = pygame.display.set_mode((WIN_W, WIN_H)) # 창 크기 설정
        pygame.display.set_caption('오목 게임') # 창 제목 설정

        # 폰트 설정
        self.font_large  = pygame.font.SysFont('malgungothic', 36, bold=True)
        self.font_medium = pygame.font.SysFont('malgungothic', 20, bold=True)
        self.font_small  = pygame.font.SysFont('malgungothic', 14)

        # 게임 상태 변수
        self.scene        = 'start'
        self.mode         = None 
        self.human_player = 1 # 인간이 사용하는 돌(흑돌)
        self.engine       = None 
        self.agent        = None 
        self.last_move    = None 

        self.clock = pygame.time.Clock() # 프레임 속도 조절용 시계

    # 메인 루프
    def run(self):
        while True: 
            self.handle_events() # 키보드, 마우스 등 이벤트 처리
            self.draw() # 현재 scene에 맞는 화면 그리기
            self.clock.tick(30) # 초당 최대 30프레임

    # 이벤트 처리
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # 창 닫기 버튼 클릭 
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN: 
                mx, my = event.pos

                if self.scene == 'start':
                    self.handle_start_click(mx, my)

                elif self.scene == 'game':
                    self.handle_game_click(mx, my)

                elif self.scene == 'gameover':
                    self.handle_gameover_click(mx, my)

    # 시작 화면 클릭
    def handle_start_click(self, mx, my):
        # 인간 vs 인간 버튼
        btn1_rect = pygame.Rect(WIN_W//2 - 130, 250, 260, 55)
        # 인간 vs AI 버튼
        btn2_rect = pygame.Rect(WIN_W//2 - 130, 330, 260, 55)

        if btn1_rect.collidepoint(mx, my): 
            self.start_human_game()

        elif btn2_rect.collidepoint(mx, my): 
            self.start_ai_game()

    # 게임 화면 클릭
    def handle_game_click(self, mx, my):
        # 처음으로 버튼
        btn_rect = pygame.Rect(
            CANVAS_SIZE + 20, WIN_H - 70, 180, 45)
        if btn_rect.collidepoint(mx, my):
            self.scene = 'start'
            return

        # 보드 클릭
        if self.engine and not self.engine.is_over: # AI 모드에서 사람 차례인지 확인
            if (self.mode == 'ai' and self.engine.current_player != self.human_player):
                return # AI모드에서 AI차례면 클릭 무시

            j = round((mx - MARGIN) / CELL_SIZE) # 클릭 픽셀 좌표 보드 좌표로 변환
            i = round((my - MARGIN) / CELL_SIZE)

            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
                if self.engine.make_move(i, j): # 착수 성공 시
                    self.last_move = (i, j) # 마지막 착수 위치 저장
                    self.check_game_over()

                    # AI 차례 실행
                    if (self.mode == 'ai' and not self.engine.is_over):
                        pygame.time.set_timer(pygame.USEREVENT, 300)

    # 게임 종료 화면 클릭
    def handle_gameover_click(self, mx, my):
        # 확인 버튼
        btn_rect = pygame.Rect(WIN_W//2 - 80, WIN_H//2 + 40, 160, 45)
        if btn_rect.collidepoint(mx, my):
            self.scene = 'start'

    # 그리기
    def draw(self):
        if self.scene == 'start':
            self.draw_start_screen()
        elif self.scene == 'game':
            self.draw_game_screen()
            # AI 이벤트 처리
            for event in pygame.event.get(pygame.USEREVENT): # # USEREVENT 발생 시 (AI 딜레이 타이머)
                self.ai_move()
                pygame.time.set_timer(pygame.USEREVENT, 0) # 타이머 초기화
        elif self.scene == 'gameover':
            self.draw_game_screen()
            self.draw_gameover_popup()

        pygame.display.flip()

    # 시작 화면 그리기
    def draw_start_screen(self):
        self.screen.fill(BG_COLOR)

        # 타이틀
        title = self.font_large.render('오목 게임', True, BLACK_COLOR) # 텍스트 렌더링
        self.screen.blit(title, (WIN_W//2 - title.get_width()//2, 130)) # 화면 중앙에 표시

        # 인간 vs 인간 버튼
        btn1_rect = pygame.Rect(WIN_W//2 - 130, 250, 260, 55)
        pygame.draw.rect(self.screen, BLACK_COLOR, btn1_rect, border_radius=2)
        t1 = self.font_medium.render('인간  vs  인간', True, WHITE_COLOR)
        self.screen.blit(t1, (WIN_W//2 - t1.get_width()//2, 265))

        # 인간 vs AI 버튼
        btn2_rect = pygame.Rect(WIN_W//2 - 130, 330, 260, 55)
        pygame.draw.rect(self.screen, WHITE_COLOR, btn2_rect, border_radius=2)
        pygame.draw.rect(self.screen, BLACK_COLOR, btn2_rect, border_radius=2, width=1)
        t2 = self.font_medium.render('인간  vs  AI', True, BLACK_COLOR)
        self.screen.blit(t2, (WIN_W//2 - t2.get_width()//2, 345))

    # 게임 화면 그리기
    def draw_game_screen(self):
        self.screen.fill(BG_COLOR)

        # 바둑판 배경으로 채우기
        pygame.draw.rect(self.screen, BOARD_COLOR, (0, 0, CANVAS_SIZE, WIN_H)) 

        # 바둑판 선 그리기
        for i in range(BOARD_SIZE):
            # 세로선
            pygame.draw.line(self.screen, LINE_COLOR,
                (MARGIN + i*CELL_SIZE, MARGIN),
                (MARGIN + i*CELL_SIZE,
                 MARGIN + (BOARD_SIZE-1)*CELL_SIZE), 1)
            # 가로선
            pygame.draw.line(self.screen, LINE_COLOR,
                (MARGIN, MARGIN + i*CELL_SIZE),
                (MARGIN + (BOARD_SIZE-1)*CELL_SIZE,
                 MARGIN + i*CELL_SIZE), 1)

        # 화점 그리기(바둑판 표준 위치)
        for sr in [3, 7, 11]:
            for sc in [3, 7, 11]:
                cx = MARGIN + sc*CELL_SIZE
                cy = MARGIN + sr*CELL_SIZE
                pygame.draw.circle(
                    self.screen, LINE_COLOR, (cx, cy), 4)

        # 금수 X 표시 (흑돌)
        if (self.engine and
                self.engine.current_player == 1 and
                not self.engine.is_over): # 게임중이고, 흑돌 차례일 때
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (self.engine.board.board[i][j] == 0 and
                            Rules.is_forbidden(
                                self.engine.board.board,
                                i, j, 1)): # 빈칸이고 금수 위치면
                        cx = MARGIN + j*CELL_SIZE
                        cy = MARGIN + i*CELL_SIZE
                        pygame.draw.line(
                            self.screen, RED_COLOR,
                            (cx-7, cy-7), (cx+7, cy+7), 2) 
                        pygame.draw.line(
                            self.screen, RED_COLOR,
                            (cx+7, cy-7), (cx-7, cy+7), 2)

        # 돌 그리기
        if self.engine:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE): # 중심 픽셀 좌표
                    cx = MARGIN + j*CELL_SIZE
                    cy = MARGIN + i*CELL_SIZE

                    if self.engine.board.board[i][j] == 1:
                        # 흑돌
                        pygame.draw.circle(self.screen, BLACK_COLOR, (cx, cy), 16)

                    elif self.engine.board.board[i][j] == 2:
                        # 백돌
                        pygame.draw.circle(self.screen, WHITE_COLOR, (cx, cy), 16)

        # 마지막 착수 (빨간 점)
        if self.last_move:
            li, lj = self.last_move # 마지막 수의 행 열
            cx     = MARGIN + lj*CELL_SIZE
            cy     = MARGIN + li*CELL_SIZE
            pygame.draw.circle(self.screen, RED_COLOR, (cx, cy), 5) 

        # 우측 패널
        self.draw_panel()

    # 우측 패널 그리기
    def draw_panel(self):
        # 패널 배경
        pygame.draw.rect(self.screen, PANEL_COLOR, (CANVAS_SIZE, 0, PANEL_W, WIN_H))

        # 타이틀
        title = self.font_medium.render('오목 게임', True, TEXT_COLOR)
        self.screen.blit(title,
            (CANVAS_SIZE + PANEL_W//2 - title.get_width()//2, 30)) # 중앙에 타이틀 표시

        # 구분선
        pygame.draw.line(self.screen, (70, 70, 70), (CANVAS_SIZE + 20, 70), (CANVAS_SIZE + PANEL_W - 20, 70), 1) 

        # 현재 차례 표시
        turn_label = self.font_small.render('현재 차례', True, GRAY_COLOR)
        self.screen.blit(turn_label, (CANVAS_SIZE + PANEL_W//2 - turn_label.get_width()//2, 100))

        if self.engine and not self.engine.is_over:
            if self.engine.current_player == 1:
                turn_text = '흑돌'
            else:
                turn_text = '백돌'
            turn = self.font_medium.render(turn_text, True, TEXT_COLOR)
            self.screen.blit(turn, (CANVAS_SIZE + PANEL_W//2 - turn.get_width()//2, 130))

        # 구분선
        pygame.draw.line(self.screen, (70, 70, 70), (CANVAS_SIZE + 20, 175), (CANVAS_SIZE + PANEL_W - 20, 175), 1)

        # 모드 표시
        mode_str = ('인간 vs 인간' if self.mode == 'human' else '인간 vs AI')
        mode_text = self.font_small.render(mode_str, True, GRAY_COLOR)
        self.screen.blit(mode_text, (CANVAS_SIZE + PANEL_W//2 - mode_text.get_width()//2, 190))

        # AI 모드 역할 표시
        if self.mode == 'ai':
            my   = self.font_small.render('나: 흑돌 ', True, GRAY_COLOR)
            ai   = self.font_small.render('AI: 백돌', True, GRAY_COLOR)
            self.screen.blit(my,(CANVAS_SIZE + PANEL_W//2 - my.get_width()//2, 215)) # 내 돌 표시
            self.screen.blit(ai,(CANVAS_SIZE + PANEL_W//2 - ai.get_width()//2, 238)) # AI 돌 표시

        # 처음으로 버튼
        btn_rect = pygame.Rect(CANVAS_SIZE + 20, WIN_H - 70, 180, 45)
        pygame.draw.rect(self.screen, BG_COLOR, btn_rect, border_radius=2)
        btn_text = self.font_small.render('처음으로', True, BLACK_COLOR)
        self.screen.blit(btn_text, (CANVAS_SIZE + PANEL_W//2 - btn_text.get_width()//2, WIN_H - 55))

    # 게임 종료 팝업
    def draw_gameover_popup(self):
        # 반투명 오버레이
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # 팝업 박스
        box_w, box_h = 320, 160
        box_x = WIN_W//2 - box_w//2
        box_y = WIN_H//2 - box_h//2

        # 팝업 배경
        pygame.draw.rect(self.screen, BG_COLOR, (box_x, box_y, box_w, box_h), border_radius=2)
        
        # 팝업 테두리 
        pygame.draw.rect(self.screen, BLACK_COLOR, (box_x, box_y, box_w, box_h), border_radius=2, width=1)

        # 제목
        title = self.font_small.render('게임종료!', True, GRAY_COLOR)
        self.screen.blit(title, (WIN_W//2 - title.get_width()//2, box_y + 20))

        # 결과 메시지
        msg = self.font_medium.render(self.gameover_msg, True, BLACK_COLOR)
        self.screen.blit(msg, (WIN_W//2 - msg.get_width()//2, box_y + 60))

        # 확인 버튼
        btn_rect = pygame.Rect(WIN_W//2 - 80, box_y + 105, 160, 40)
        pygame.draw.rect(self.screen, BLACK_COLOR, btn_rect)
        btn_text = self.font_small.render('확 인', True, BG_COLOR)
        self.screen.blit(btn_text, (WIN_W//2 - btn_text.get_width()//2, box_y + 117))

    # AI 착수
    def ai_move(self):
        if self.agent is None or self.engine.is_over: # AI 없거나, 게임 종료 시 
            return
        if self.engine.current_player != self.agent.player: # AI 차례가 아닐 때
            return

        move = self.agent.decide_next_move(self.engine) # AI가 다음 착수 위치 결정
        if move:
            self.engine.make_move(*move) # 결정된 위치에 착수, *move : (행, 열) 튜플을 개별 인자로 전달
            self.last_move = move # 마지막 착수 위치 저장(빨간점)
            reward = self.agent.calculate_reward(self.engine) 
            self.agent.store_reward(reward)
            self.check_game_over()

    # 게임 종료 확인
    def check_game_over(self):
        if not self.engine.is_over:
            return

        if self.engine.winner == 1:
            self.gameover_msg = '흑돌 승리!'
        elif self.engine.winner == 2:
            self.gameover_msg = '백돌 승리!'
        else:
            self.gameover_msg = '무승부!'

        self.scene = 'gameover'

    # 게임 시작
    def start_human_game(self):
        self.mode      = 'human' 
        self.engine    = Engine(BOARD_SIZE)
        self.agent     = None
        self.last_move = None
        self.scene     = 'game'

    def start_ai_game(self):
        self.mode         = 'ai'
        self.human_player = 1
        self.engine       = Engine(BOARD_SIZE)
        self.last_move    = None
        self.agent        = PPOAgent(BOARD_SIZE, player=2)
        self.scene        = 'game'
