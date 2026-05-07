import numpy as np

class Board:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """보드 초기화"""
        # 0: 빈칸, 1: 흑돌, 2: 백돌
        self.board          = np.zeros(
            (self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 흑돌 선공
        self.is_over        = False
        self.winner         = None
        return self.get_state()

    def get_state(self):
        """CNN 입력용 상태 변환 (1, board_size, board_size)"""
        return self.board.copy()\
            .reshape(1, self.board_size, self.board_size)\
            .astype(np.float32)

    def get_valid_moves(self):
        """빈 칸 목록 반환"""
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        """착수 처리"""
        if self.board[row, col] != 0 or self.is_over:
            return False

        self.board[row, col] = self.current_player

        # 승리 판정
        from core.Rules import Rules
        if Rules.check_win(self.board, row, col):
            self.is_over = True
            self.winner  = self.current_player

        # 무승부 판정
        elif not np.any(self.board == 0):
            self.is_over = True
            self.winner  = 0

        # 턴 교체 1→2, 2→1
        self.current_player = 3 - self.current_player
        return True