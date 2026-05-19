# engine.py
from core.Board import Board
from core.Rules import Rules


class Engine:

    def __init__(self, board_size=15):

        self.board_size = board_size
        self.board = Board(board_size)

    def reset(self):

        self.board.reset()
        return self.get_state()

    def get_state(self):

        return self.board.get_state()

    def get_valid_moves(self):

        return list(self.board.get_valid_moves())

    def make_move(self, row, col):

        if (
            row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        ):
            return False

        # Board.make_move()가 승리·무승부 판정을 모두 처리하므로
        success = self.board.make_move(row, col)

        return success

    def step(self, row, col):

        success = self.make_move(row, col)
        state   = self.get_state()
        done    = self.is_over
        winner  = self.winner

        return success, state, done, winner

    def check_win(self, row, col):

        return Rules.check_win(
            self.board.board,
            row,
            col
        )

    def check_patterns(self, player, length):

        return Rules.check_patterns(
            self.board.board,
            player,
            length
        )

    @property
    def current_player(self):
        return self.board.current_player

    @property
    def turn(self):
        return self.board.current_player

    @property
    def is_over(self):
        return self.board.is_over

    @property
    def winner(self):
        w = self.board.winner
        if w is None:
            return 0
        return int(w)