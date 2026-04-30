import random
from core.omoku_rules import EMPTY, BLACK, WHITE

class OmokuAI:
    def __init__(self, player=WHITE, difficulty="random"):
        self.player = player
        self.difficulty = difficulty

    def get_move(self, board):
        size = len(board)
        empty_cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if board[r][c] == EMPTY
        ]
        if not empty_cells:
            return (size // 2, size // 2)
        return random.choice(empty_cells)
