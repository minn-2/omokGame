import numpy as np

class Rules:
    DIRECTIONS = [
        (0, 1), (1, 0), (1, 1), (1, -1)
    ]

    @staticmethod
    def check_win(board, r, c):
        player = board[r, c]
        board_size = len(board)
        if player == 0: return False

        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player):
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            if player == 1 and count == 5: return True
            if player == 2 and count >= 5: return True
        return False

    # [수정] 금수 체크 로직 분리 (재귀 방지용 check_only 추가)
    @staticmethod
    def is_forbidden(board, r, c, player, check_only=False):
        if player != 1: return False
        if board[r, c] != 0: return True

        board[r, c] = player
        if Rules.check_win(board, r, c):
            board[r, c] = 0
            return False

        # check_only가 True면 재귀를 타지 않고 단순 금수 조건만 확인
        forbidden = False
        if Rules._is_overline(board, r, c, player):
            forbidden = True
        elif Rules._count_legal_fours(board, r, c, player) >= 2:
            forbidden = True
        elif not check_only and Rules._count_legal_threes(board, r, c, player) >= 2:
            forbidden = True

        board[r, c] = 0
        return forbidden

    @staticmethod
    def _is_overline(board, r, c, player):
        board_size = len(board)
        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player):
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            if count >= 6: return True
        return False

    @staticmethod
    def _count_legal_fours(board, r, c, player):
        board_size = len(board)
        four_count = 0
        for dr, dc in Rules.DIRECTIONS:
            found = False
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr, nc] != 0: continue
                board[nr, nc] = player
                if Rules.check_win(board, nr, nc): found = True
                board[nr, nc] = 0
                if found: break
            if found: four_count += 1
        return four_count

    @staticmethod
    def _count_legal_threes(board, r, c, player):
        board_size = len(board)
        three_count = 0
        for dr, dc in Rules.DIRECTIONS:
            is_open_three = False
            for i in range(-4, 5):
                if i == 0: continue
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr, nc] != 0: continue
                
                board[nr, nc] = player
                if not Rules._is_overline(board, nr, nc, player):
                    win_spots = 0
                    for j in range(-4, 5):
                        nnr, nnc = nr + j * dr, nc + j * dc
                        if not (0 <= nnr < board_size and 0 <= nnc < board_size) or board[nnr, nnc] != 0: continue
                        board[nnr, nnc] = player
                        if Rules.check_win(board, nnr, nnc): win_spots += 1
                        board[nnr, nnc] = 0
                    
                    if win_spots >= 2:
                        board[nr, nc] = 0
                        # [핵심] 여기서 check_only=True로 호출하여 재귀 루프를 차단합니다!
                        if not Rules.is_forbidden(board, nr, nc, player, check_only=True):
                            is_open_three = True
                        board[nr, nc] = player
                board[nr, nc] = 0
                if is_open_three: break
            if is_open_three: three_count += 1
        return three_count
