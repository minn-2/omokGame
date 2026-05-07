from core.Board import Board
from core.Rules import Rules

class Engine:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board      = Board(board_size)

    def reset(self):
        """게임 초기화"""
        return self.board.reset()

    def get_state(self):
        """CNN 입력용 상태 반환"""
        return self.board.get_state()

    def get_valid_moves(self):
        """유효한 착수 위치 반환"""
        return self.board.get_valid_moves()

    def make_move(self, row, col):
        """착수"""
        return self.board.make_move(row, col)

    def check_win(self, r, c):
        """승리 판정"""
        return Rules.check_win(self.board.board, r, c)

    def check_patterns(self, player, length):
        """패턴 감지 (3목 / 4목)"""
        return Rules.check_patterns(self.board.board, player, length)

    @property
    def current_player(self):
        return self.board.current_player

    @property
    def is_over(self):
        return self.board.is_over

    @property
    def winner(self):
        return self.board.winner