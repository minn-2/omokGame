import numpy as np

class Board:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board          = np.zeros(
            (self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 흑돌 선공
        self.is_over        = False
        self.winner         = None
        return self.get_state()

    def get_state(self):
        return self.board.copy()\
            .reshape(1, self.board_size, self.board_size)\
            .astype(np.float32)

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):  # ← 들여쓰기 수정 (클래스 안으로)
        if self.board[row, col] != 0 or self.is_over:
            return False

        from core.Rules import Rules

        if Rules.is_forbidden(
                self.board, row, col, self.current_player):
            return False

        self.board[row, col] = self.current_player

        if Rules.check_win(self.board, row, col):
            self.is_over = True
            self.winner  = self.current_player

        elif not np.any(self.board == 0):
            self.is_over = True
            self.winner  = 0

        self.current_player = 3 - self.current_player
        return True