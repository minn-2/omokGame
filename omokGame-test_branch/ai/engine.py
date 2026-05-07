import numpy as np

class OmokEngine:
    def __init__(self, board_size=15):
        """
        보드 크기 (기본 15x15)
        """
        self.board_size = board_size
        # 실제 돌이 놓일 보드 배열 선언 (0: 빈칸, 1: 흑돌, 2: 백돌)
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.is_over = False
        self.winner = 0

    def reset(self):
        """
        게임 상태 초기화: 새로운 대국을 시작할 때 호출 
        """
        # 불필요한 float 연산을 방지하고 메모리 점유율 최적화 함
        self.board.fill(0)
        self.current_player = 1 # 흑돌(1) 선공 원칙
        self.last_move = None
        self.is_over = False # 게임 종료 플래그 
        self.winner = 0 # 승자 상태 (1,2,0=Draw)

        # 초기 관측값을 반환하여 에이전트가 학습을 시작함 
        return self.get_state()

    def get_state(self):
        """
        현재 보드 상태를 CNN 모델(Neural Network)이 처리할 수 있는
        3D float32 텐서 형태로 변환 (Shape: 1, Size, Size)
        """
        # copy()사용하는 이유: 원본 보드 데이터 보호하기 위해서 
        return self.board.copy().reshape(1, self.board_size, self.board_size).astype(np.float32)

    def get_valid_moves(self):
        """
        AI가 이미 돌이 있는 곳에 두지 않도록 선택지를 제한하는 역할 
        """
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        """
        에이전트의 행동을 받아 다음 상태로 보드 업데이트
        """
        # 이미 돌이 있거나 게임이 종료된 경우 거부 
        if self.board[row, col] != 0 or self.is_over:
            return False
        
        # 돌 배치
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # 승리 조건 검사
        if self.check_win(row, col):
            self.is_over = True
            self.winner = self.current_player

        # 무승부 검사 (보드가 꽉 찼는지)
        elif not np.any(self.board == 0):
            self.is_over = True
            self.winner = 0 # 0은 무승부
            
        # 플레이어 교체: 1 -> 2 / 2 -> 1
        self.current_player = 3 - self.current_player
        return True

    def check_win(self, r, c):
        """
        방금 둔 돌을 기점으로 8방향을 검사해서 5연속 여부 확인 
        """
        player = self.board[r, c]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            
            if count >= 5: return True
        return False

    def check_patterns(self, player, length):
        """
        AI 학습 가속화를 위한 3목, 4목 패턴 전수 조사
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == player:
                    for dr, dc in directions:
                        count = 1
                        nr, nc = r + dr, c + dc
                        while 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                            count += 1
                            nr += dr
                            nc += dc
                        if count == length:
                            return True
        return False
