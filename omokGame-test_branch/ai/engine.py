from core.Board import Board
from core.Rules import Rules

class Engine:
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.board      = Board(board_size)
        self._sync()

    # 내부 동기화
    def _sync(self):
        self.current_player = self.board.current_player
        self.is_over        = self.board.is_over
        self.winner         = self.board.winner

    # 공개 인터페이스
    def make_move(self, row: int, col: int) -> bool:
        result = self.board.make_move(row, col)
        self._sync()
        return result

    def get_state(self):
        return self.board.get_state()

    def get_valid_moves(self):
        return self.board.get_valid_moves()

    def reset(self):
        self.board.reset()
        self._sync()
        return self.get_state()

    # 편의 프로퍼티
    @property
    def numpy_board(self):
        return self.board.board