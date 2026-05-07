import pygame
import os
import sys
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from core.Board  import Board
from ai.agent    import PPOAgent
from ai.engine   import Engine

# 색상
BLACK     = (0,   0,   0  )
WHITE     = (255, 255, 255)
BROWN     = (185, 122, 87 )
BTN_BLACK = (50,  50,  50 )
BTN_BLUE  = (70,  130, 180)

# 화면 설정
CELL_SIZE  = 40
BOARD_SIZE = 15
MARGIN     = 40
SCREEN_W   = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
SCREEN_H   = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2 + 60


class Play:
    def __init__(self):
        pygame.init()
        self.screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption('오목 게임')
        self.font     = pygame.font.SysFont('malgungothic', 20)
        self.big_font = pygame.font.SysFont('malgungothic', 30)

        # 게임 상태
        self.scene        = 'start'  # start → select → game
        self.mode         = None     # human / ai
        self.human_player = 1        # 1=흑돌, 2=백돌
        self.engine       = None
        self.agent        = None

    # ─────────────────── 화면 그리기

    def draw_start_screen(self):
        """시작 화면"""
        self.screen.fill(BROWN)

        title = self.big_font.render('오목 게임', True, BLACK)
        self.screen.blit(title,
            (SCREEN_W//2 - title.get_width()//2, 150))

        # 인간 vs 인간
        pygame.draw.rect(self.screen, BTN_BLACK,
            (SCREEN_W//2-120, 250, 240, 60), border_radius=10)
        t1 = self.font.render('인간 vs 인간', True, WHITE)
        self.screen.blit(t1,
            (SCREEN_W//2 - t1.get_width()//2, 268))

        # 인간 vs AI
        pygame.draw.rect(self.screen, BTN_BLUE,
            (SCREEN_W//2-120, 340, 240, 60), border_radius=10)
        t2 = self.font.render('인간 vs AI', True, WHITE)
        self.screen.blit(t2,
            (SCREEN_W//2 - t2.get_width()//2, 358))

        pygame.display.flip()

    def draw_select_screen(self):
        """흑돌 / 백돌 선택 화면"""
        self.screen.fill(BROWN)

        title = self.big_font.render('돌 색상 선택', True, BLACK)
        self.screen.blit(title,
            (SCREEN_W//2 - title.get_width()//2, 150))

        sub = self.font.render('(흑돌이 선공입니다)', True, BTN_BLACK)
        self.screen.blit(sub,
            (SCREEN_W//2 - sub.get_width()//2, 200))

        # 흑돌 버튼
        pygame.draw.circle(self.screen, BLACK,
            (SCREEN_W//2-80, 300), 35)
        t1 = self.font.render('흑돌', True, WHITE)
        self.screen.blit(t1,
            (SCREEN_W//2-80 - t1.get_width()//2, 345))

        # 백돌 버튼
        pygame.draw.circle(self.screen, WHITE,
            (SCREEN_W//2+80, 300), 35)
        pygame.draw.circle(self.screen, BLACK,
            (SCREEN_W//2+80, 300), 35, 2)
        t2 = self.font.render('백돌', True, BLACK)
        self.screen.blit(t2,
            (SCREEN_W//2+80 - t2.get_width()//2, 345))

        pygame.display.flip()

    def draw_board(self):
        """게임 보드 화면"""
        self.screen.fill(BROWN)

        # 바둑판 선
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, BLACK,
                (MARGIN + i*CELL_SIZE, MARGIN),
                (MARGIN + i*CELL_SIZE,
                 MARGIN + (BOARD_SIZE-1)*CELL_SIZE), 1)
            pygame.draw.line(self.screen, BLACK,
                (MARGIN, MARGIN + i*CELL_SIZE),
                (MARGIN + (BOARD_SIZE-1)*CELL_SIZE,
                 MARGIN + i*CELL_SIZE), 1)

        # 돌 그리기
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.engine.board.board[i][j] == 1:
                    pygame.draw.circle(self.screen, BLACK,
                        (MARGIN + j*CELL_SIZE,
                         MARGIN + i*CELL_SIZE), 18)
                elif self.engine.board.board[i][j] == 2:
                    pygame.draw.circle(self.screen, WHITE,
                        (MARGIN + j*CELL_SIZE,
                         MARGIN + i*CELL_SIZE), 18)
                    pygame.draw.circle(self.screen, BLACK,
                        (MARGIN + j*CELL_SIZE,
                         MARGIN + i*CELL_SIZE), 18, 1)

        # 상태 표시
        if self.engine.is_over:
            if self.engine.winner == 0:
                status = '무승부!'
            elif self.engine.winner == 1:
                status = '흑돌 승리!'
            else:
                status = '백돌 승리!'
        else:
            status = '흑돌 차례' \
                if self.engine.current_player == 1 \
                else '백돌 차례'

        self.screen.blit(
            self.font.render(status, True, BLACK),
            (MARGIN, SCREEN_H-50))

        # 리셋 버튼
        pygame.draw.rect(self.screen, BTN_BLACK,
            (SCREEN_W-120, SCREEN_H-55, 100, 40),
            border_radius=8)
        reset_text = self.font.render('Reset', True, WHITE)
        self.screen.blit(reset_text,
            (SCREEN_W-70 - reset_text.get_width()//2,
             SCREEN_H-45))

        pygame.display.flip()

    # ─────────────────── 좌표 변환

    def get_board_pos(self, mx, my):
        j = round((mx - MARGIN) / CELL_SIZE)
        i = round((my - MARGIN) / CELL_SIZE)
        if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE:
            return i, j
        return None, None

    # ─────────────────── 게임 시작 / 리셋

    def start_game(self):
        self.engine = Engine(BOARD_SIZE)
        if self.mode == 'ai':
            ai_player  = 3 - self.human_player
            self.agent = PPOAgent(BOARD_SIZE, player=ai_player)

    def reset_game(self):
        self.scene        = 'start'
        self.mode         = None
        self.human_player = 1
        self.engine       = None
        self.agent        = None

    # ─────────────────── AI 착수

    def ai_move(self):
        if self.agent is None or self.engine.is_over:
            return
        if self.engine.current_player != self.agent.player:
            return
        move = self.agent.decide_next_move(self.engine)
        if move:
            self.engine.make_move(*move)
            reward = self.agent.calculate_reward(self.engine)
            self.agent.store_reward(reward)

    # ─────────────────── 메인 루프

    def run(self):
        clock = pygame.time.Clock()

        while True:

            # 시작 화면
            if self.scene == 'start':
                self.draw_start_screen()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = event.pos
                        # 인간 vs 인간
                        if (SCREEN_W//2-120 <= mx <= SCREEN_W//2+120
                                and 250 <= my <= 310):
                            self.mode  = 'human'
                            self.scene = 'game'
                            self.start_game()
                        # 인간 vs AI
                        elif (SCREEN_W//2-120 <= mx <= SCREEN_W//2+120
                                and 340 <= my <= 400):
                            self.mode  = 'ai'
                            self.scene = 'select'

            # 돌 색상 선택 화면
            elif self.scene == 'select':
                self.draw_select_screen()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = event.pos
                        # 흑돌 선택
                        if ((mx-(SCREEN_W//2-80))**2
                                + (my-300)**2 <= 35**2):
                            self.human_player = 1
                            self.scene        = 'game'
                            self.start_game()
                        # 백돌 선택
                        elif ((mx-(SCREEN_W//2+80))**2
                                + (my-300)**2 <= 35**2):
                            self.human_player = 2
                            self.scene        = 'game'
                            self.start_game()

            # 게임 화면
            elif self.scene == 'game':
                # AI 차례 자동 착수
                if self.mode == 'ai' and not self.engine.is_over:
                    self.ai_move()

                self.draw_board()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mx, my = event.pos
                        # 리셋 버튼
                        if (SCREEN_W-120 <= mx <= SCREEN_W-20
                                and SCREEN_H-55 <= my <= SCREEN_H-15):
                            self.reset_game()
                        # 보드 클릭
                        else:
                            i, j = self.get_board_pos(mx, my)
                            if i is not None and not self.engine.is_over:
                                if self.mode == 'human':
                                    self.engine.make_move(i, j)
                                elif (self.mode == 'ai' and
                                      self.engine.current_player
                                      == self.human_player):
                                    self.engine.make_move(i, j)

            clock.tick(30)


# main
if __name__ == '__main__':
    import sys
    import os
    # 프로젝트 루트 경로 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    game = Play()
    game.run()