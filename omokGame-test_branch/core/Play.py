import tkinter as tk
from tkinter import messagebox

from core.Board import Board
from core.Rules import Rules
from ai.engine  import Engine

# 색상
BG_COLOR    = '#FFFAE9'
BOARD_COLOR = '#DCB468'
LINE_COLOR  = '#8B6914'
BLACK_COLOR = '#000000'
WHITE_COLOR = '#FFFFFF'

# 화면 및 보드 설정
CELL_SIZE   = 38
BOARD_SIZE  = 15
MARGIN      = 30
CANVAS_SIZE = CELL_SIZE * (BOARD_SIZE - 1) + MARGIN * 2 # 캔버스 실제 크기
# 전체 창 크기
WIN_W       = 805
WIN_H       = 552


class Play:
    def __init__(self):
        self.window = tk.Tk() # Tkinter 메인 윈도우 생성
        self.window.title('오목 게임') # 창 제목 설정
        self.window.resizable(False, False) # 창 크기 고정
        self.window.configure(bg=BG_COLOR) # 배경색 설정

        self.mode      = None # 게임 모드 저장
        self.engine    = None  # 게임 엔진 저장
        self.last_move = None # 마지막으로 둔 돌 위치 저장

        self.show_start_screen() # 시작 화면 표시

    # 시작 화면
    def show_start_screen(self):
        # 기존 위젯 모두 제거
        for w in self.window.winfo_children():
            w.destroy()

        self.window.geometry(f'{WIN_W}x{WIN_H}') # 창 크기 설정
        self.window.configure(bg=BG_COLOR)

        center = tk.Frame(self.window, bg=BG_COLOR)
        center.place(relx=0.5, rely=0.5, anchor='center')

        # 타이틀
        tk.Label(center,
            text='오목 게임',
            font=('맑은 고딕', 36, 'bold'),
            bg=BG_COLOR, fg=BLACK_COLOR
        ).pack(pady=(0, 40))

        # 인간 vs 인간 버튼
        tk.Button(center,
            text='인간  vs  인간',
            font=('맑은 고딕', 14),
            bg=BLACK_COLOR, fg=BG_COLOR,
            activebackground='#333333',
            activeforeground=BG_COLOR,
            width=16, height=2,
            relief='flat', cursor='hand2',
            command=self.start_human_game # 버튼 클릭 시 실행
        ).pack(pady=8)

        # 인간 vs AI 버튼
        tk.Button(center,
            text='인간  vs  AI',
            font=('맑은 고딕', 14),
            bg=WHITE_COLOR, fg=BLACK_COLOR,
            activebackground='#EEEEEE',
            activeforeground=BLACK_COLOR,
            width=16, height=2,
            relief='solid', cursor='hand2',
            bd=1,
            command=self.start_ai_game # AI 모드 시작
        ).pack(pady=8) 

    # 게임 화면
    def show_game_screen(self):
        for w in self.window.winfo_children():
            w.destroy()

        self.window.geometry(f'{WIN_W}x{WIN_H}')
        self.window.configure(bg=BG_COLOR)

        # 왼쪽: 바둑판
        left = tk.Frame(self.window, bg=BG_COLOR)
        left.pack(side='left', padx=10, pady=10)

        self.canvas = tk.Canvas(left,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BOARD_COLOR,
            highlightthickness=2,
            highlightbackground=LINE_COLOR)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_click)

        # 오른쪽: 정보 패널
        right = tk.Frame(self.window,
            bg=BLACK_COLOR, width=240)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

        # 타이틀
        tk.Label(right,
            text='오목 게임',
            font=('맑은 고딕', 20, 'bold'),
            bg=BLACK_COLOR, fg=BG_COLOR,
            pady=20
        ).pack(fill='x')

        # 구분선
        tk.Frame(right, bg='#444444',
            height=1).pack(fill='x', padx=20)

        # 차례 표시
        turn_frame = tk.Frame(right, bg=BLACK_COLOR)
        turn_frame.pack(pady=40)

        tk.Label(turn_frame,
            text='현재 차례',
            font=('맑은 고딕', 11),
            bg=BLACK_COLOR, fg='#AAAAAA'
        ).pack(pady=(0, 10))

        # 현재 돌 표시
        self.turn_label = tk.Label(turn_frame,
            text='흑돌',
            font=('맑은 고딕', 20, 'bold'),
            bg=BLACK_COLOR, fg=WHITE_COLOR)
        self.turn_label.pack()

        # 구분선
        tk.Frame(right, bg='#444444',
            height=1).pack(fill='x', padx=20, pady=20)

        # 모드 표시
        mode_str = ('인간 vs 인간'
            if self.mode == 'human' else '인간 vs AI')
        tk.Label(right,
            text=mode_str,
            font=('맑은 고딕', 11),
            bg=BLACK_COLOR, fg='#AAAAAA'
        ).pack()

        # 처음으로 버튼
        tk.Button(right,
            text='처음으로',
            font=('맑은 고딕', 12),
            bg=BG_COLOR, fg=BLACK_COLOR,
            activebackground='#EEEEEE',
            relief='flat', cursor='hand2',
            pady=8, width=12,
            command=self.show_start_screen
        ).pack(side='bottom', pady=30)

        self.draw_board()

    # 보드 그리기
    def draw_board(self):
        # 기존 그림 삭제
        self.canvas.delete('all')

        # 바둑판 선
        for i in range(BOARD_SIZE):
            self.canvas.create_line( # 세로줄
                MARGIN + i*CELL_SIZE, MARGIN, 
                MARGIN + i*CELL_SIZE,
                MARGIN + (BOARD_SIZE-1)*CELL_SIZE,
                fill=LINE_COLOR, width=1)
            self.canvas.create_line( # 가로줄
                MARGIN, MARGIN + i*CELL_SIZE,
                MARGIN + (BOARD_SIZE-1)*CELL_SIZE,
                MARGIN + i*CELL_SIZE,
                fill=LINE_COLOR, width=1)

        # 화점
        for sr in [3, 7, 11]:
            for sc in [3, 7, 11]:
                cx = MARGIN + sc*CELL_SIZE
                cy = MARGIN + sr*CELL_SIZE
                self.canvas.create_oval(
                    cx-4, cy-4, cx+4, cy+4,
                    fill=LINE_COLOR, outline='')

        # 금수 X 표시 (흑돌 차례)
        if (self.engine.current_player == 1
                and not self.engine.is_over):
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if (self.engine.board.board[i][j] == 0 and # 빈 칸이고 금수이면
                            Rules.is_forbidden(
                                self.engine.board.board,
                                i, j, 1)):
                        cx = MARGIN + j*CELL_SIZE
                        cy = MARGIN + i*CELL_SIZE
                        self.canvas.create_line( # x 표시
                            cx-7, cy-7, cx+7, cy+7,
                            fill='red', width=2)
                        self.canvas.create_line(
                            cx+7, cy-7, cx-7, cy+7,
                            fill='red', width=2)

        # 돌 그리기
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                cx = MARGIN + j*CELL_SIZE
                cy = MARGIN + i*CELL_SIZE

                if self.engine.board.board[i][j] == 1:
                    # 흑돌
                    self.canvas.create_oval(
                        cx-16, cy-16, cx+16, cy+16,
                        fill='black', outline='black')

                elif self.engine.board.board[i][j] == 2:
                    # 백돌
                    self.canvas.create_oval(
                        cx-16, cy-16, cx+16, cy+16,
                        fill='white', outline='black', width=2)

        # 마지막 착수 (빨간 점)
        if self.last_move:
            li, lj = self.last_move
            cx     = MARGIN + lj*CELL_SIZE
            cy     = MARGIN + li*CELL_SIZE
            self.canvas.create_oval(
                cx-5, cy-5, cx+5, cy+5,
                fill='red', outline='')
        
        # 차례 텍스트 업데이트 호출 
        self.update_turn_display()

    # 차례 텍스트 업데이트
    def update_turn_display(self):
        # 게임 종료 시 변경 안 함
        if self.engine.is_over:
            return
        # 현재 플레이어 표시
        if self.engine.current_player == 1:
            self.turn_label.config(text='흑돌')
        else:
            self.turn_label.config(text='백돌')

    # 클릭 이벤트
    def on_click(self, event):
        # 게임 종료 시 입력 무시
        if self.engine.is_over:
            return

        # 클릭 위치를 보드 좌표로 변환
        j = round((event.x - MARGIN) / CELL_SIZE)
        i = round((event.y - MARGIN) / CELL_SIZE)

        # 범위 밖 클릭 방지
        if not (0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE):
            return

        # 돌 놓기 성공 시
        if self.engine.make_move(i, j):
            self.last_move = (i, j) # 마지막 수 저장
            self.draw_board() # 보드 다시 그리기
            self.check_game_over() # 게임 종료 확인

    # 게임 종료
    def check_game_over(self):

        # 게임 안 끝났으면 종료
        if not self.engine.is_over:
            return

        # 메시지
        if self.engine.winner == 1:
            msg = '흑돌 승리!'
        elif self.engine.winner == 2:
            msg = '백돌 승리!'
        else:
            msg = '무승부!'

        self.turn_label.config(text=msg) # 오른쪽 패널 메시지 변경
        messagebox.showinfo('게임 종료', msg) # 팝업 출력
        self.show_start_screen() # 시작 화면으로 이동

    # 게임 시작 (인간 vs 인간)
    def start_human_game(self):
        self.mode      = 'human'
        self.engine    = Engine(BOARD_SIZE) # 게임 엔진 생성
        self.last_move = None # 마지막 수 초기화
        self.show_game_screen() # 게임 화면 표시

    # 게임 시작 (인간 vs AI)
    def start_ai_game(self):
        messagebox.showinfo('안내', '추후 구현 예정입니다!')

    # 실행
    def run(self):
        self.window.mainloop()