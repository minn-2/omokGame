# ============================================================
# gomoku_game.py  —  게임 로직 모듈 (Human vs Human)
# ============================================================
# 담당 역할
#   - 보드 상태 관리 (초기화, 복사)
#   - 착수 처리 (유효성 검사 → 금수 검사 → 승리 검사)
#   - 무르기(undo) 지원
#   - GUI 없이 콘솔에서도 단독 실행 가능
# ============================================================

from core.omoku_rules import EMPTY, BLACK, WHITE, check_win, is_forbidden
DEFAULT_SIZE = 15   # 표준 오목판 크기


class GomokuGame:
    """오목 게임 상태 및 규칙을 관리하는 클래스."""

    def __init__(self, size: int = DEFAULT_SIZE, use_forbidden: bool = True):
        """
        Parameters
        ----------
        size : int
            보드 크기 (size × size). 기본값 15.
        use_forbidden : bool
            흑의 금수 규칙 적용 여부. 기본값 True.
        """
        self.size = size
        self.use_forbidden = use_forbidden
        self.reset()

    # ── 게임 초기화 ──────────────────────────────────────────

    def reset(self):
        """게임을 초기 상태로 리셋한다."""
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.current_player = BLACK      # 흑 선공
        self.winner = None               # None | BLACK | WHITE
        self.game_over = False
        self.move_history = []           # [(row, col, player), ...]
        self.forbidden_last = False      # 마지막 착수가 금수였는지
        self.forbidden_reason = ""

    # ── 착수 ────────────────────────────────────────────────

    def place_stone(self, row: int, col: int) -> dict:
        """
        (row, col)에 현재 플레이어의 돌을 놓는다.

        Returns
        -------
        dict
            {
              "success": bool,
              "message": str,       # 실패 이유 or 결과 메시지
              "winner": None|BLACK|WHITE,
              "forbidden": bool,
              "forbidden_reason": str,
            }
        """
        result = {
            "success": False,
            "message": "",
            "winner": None,
            "forbidden": False,
            "forbidden_reason": "",
        }

        if self.game_over:
            result["message"] = "게임이 이미 종료되었습니다."
            return result

        # 범위 검사
        if not (0 <= row < self.size and 0 <= col < self.size):
            result["message"] = "보드 범위를 벗어났습니다."
            return result

        # 빈칸 검사
        if self.board[row][col] != EMPTY:
            result["message"] = "이미 돌이 놓인 자리입니다."
            return result

        # 임시 착수
        self.board[row][col] = self.current_player

        # 금수 검사 (흑 전용)
        if self.use_forbidden and self.current_player == BLACK:
            forbidden, reason = is_forbidden(self.board, row, col)
            if forbidden:
                self.board[row][col] = EMPTY   # 되돌리기
                result["message"] = f"금수입니다! ({reason})"
                result["forbidden"] = True
                result["forbidden_reason"] = reason
                return result

        # 착수 확정
        self.move_history.append((row, col, self.current_player))

        # 승리 판정
        if check_win(self.board, row, col, self.current_player):
            self.winner = self.current_player
            self.game_over = True
            color = "흑" if self.current_player == BLACK else "백"
            result["message"] = f"{color}이 승리했습니다! 🎉"
            result["winner"] = self.winner
        else:
            # 무승부 판정 (보드 꽉 참)
            if self._is_board_full():
                self.game_over = True
                result["message"] = "무승부! 보드가 꽉 찼습니다."
            else:
                color = "흑" if self.current_player == BLACK else "백"
                result["message"] = f"{color} 착수 완료 ({row+1}, {col+1})"

        result["success"] = True
        # 턴 교체
        if not self.game_over:
            self._switch_player()

        return result

    # ── 무르기 ───────────────────────────────────────────────

    def undo(self) -> bool:
        """
        마지막 착수를 취소한다.
        게임이 종료된 상태에서도 무르기 가능.
        반환: 성공 여부 (기록이 없으면 False)
        """
        if not self.move_history:
            return False

        row, col, player = self.move_history.pop()
        self.board[row][col] = EMPTY
        self.current_player = player   # 방금 뒀던 플레이어로 복귀
        self.winner = None
        self.game_over = False
        return True

    # ── 유틸리티 ─────────────────────────────────────────────

    def _switch_player(self):
        self.current_player = WHITE if self.current_player == BLACK else BLACK

    def _is_board_full(self) -> bool:
        return all(
            self.board[r][c] != EMPTY
            for r in range(self.size)
            for c in range(self.size)
        )

    def get_board_copy(self):
        return [row[:] for row in self.board]

    def current_player_name(self) -> str:
        return "흑(●)" if self.current_player == BLACK else "백(○)"

    # ── 콘솔 출력 (테스트용) ─────────────────────────────────

    def print_board(self):
        symbols = {EMPTY: "·", BLACK: "●", WHITE: "○"}
        header = "   " + " ".join(f"{c+1:2}" for c in range(self.size))
        print(header)
        for r in range(self.size):
            row_str = " ".join(symbols[self.board[r][c]] for c in range(self.size))
            print(f"{r+1:2} {row_str}")
        print()


# ── 콘솔 모드 단독 실행 예시 ─────────────────────────────────

if __name__ == "__main__":
    game = GomokuGame(size=15, use_forbidden=True)
    print("=== 오목 콘솔 모드 ===")
    print("입력 형식: 행 열  (예: 8 8)  |  'u' 무르기  |  'q' 종료\n")

    while not game.game_over:
        game.print_board()
        print(f"현재 차례: {game.current_player_name()}")
        cmd = input("착수 > ").strip()

        if cmd.lower() == 'q':
            print("게임을 종료합니다.")
            break
        if cmd.lower() == 'u':
            if game.undo():
                print("무르기 완료.\n")
            else:
                print("무를 수 있는 수가 없습니다.\n")
            continue

        parts = cmd.split()
        if len(parts) != 2:
            print("입력 형식이 잘못되었습니다. 예: 8 8\n")
            continue

        try:
            row, col = int(parts[0]) - 1, int(parts[1]) - 1
        except ValueError:
            print("숫자를 입력하세요.\n")
            continue

        result = game.place_stone(row, col)
        print(result["message"], "\n")

    if game.winner:
        game.print_board()
