from core.Board import Board
from core.Rules import Rules


class Engine:

    def __init__(self, board_size=15):

        self.board_size = board_size

        self.board = Board(board_size)

    # 게임 초기화
    def reset(self):

        self.board.reset()

        return self.get_state()

    # 현재 상태 반환
    def get_state(self):

        return self.board.get_state()

    # 가능한 착수 위치 반환
    def get_valid_moves(self):

        return list(
            self.board.get_valid_moves()
        )

    # 착수
    def make_move(self, row, col):

        # 범위 검사
        if (
            row < 0
            or row >= self.board_size
            or col < 0
            or col >= self.board_size
        ):
            return False

        success = self.board.make_move(
            row,
            col
        )

        return success

    # 승리 판정
    def check_win(self, row, col):

        return Rules.check_win(
            self.board.board,
            row,
            col
        )

    # 패턴 검사
    def check_patterns(self, player, length):

        return Rules.check_patterns(
            self.board.board,
            player,
            length
        )

    # 현재 플레이어 반환
    @property
    def current_player(self):

        return self.board.current_player

    # 게임 종료 여부 반환
    @property
    def is_over(self):

        return self.board.is_over
    
     # 승자 반환
    @property
    def winner(self):

        return (
            self.board.winner
            if self.board.winner is not None
            else 0
        )
