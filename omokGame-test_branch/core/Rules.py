import numpy as np

class Rules:
    # 4방향 (양방향으로 8방향 커버)
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    # 승리 판정
    @staticmethod
    def check_win(board, r, c):
        player     = board[r, c]
        board_size = len(board)

        for dr, dc in Rules.DIRECTIONS:
            count = 1

            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    count += 1
                    nr    += dr * sign
                    nc    += dc * sign

            if player == 1 and count == 5:
                return True
            if player == 2 and count >= 5:
                return True

        return False
    
    # 금수 판정 (흑돌 전용)
    @staticmethod
    def is_forbidden(board, r, c, player):
        if player != 1:
            return False

        board[r, c] = player

        # 5목이면 금수 해제
        if Rules.check_win(board, r, c):
            board[r, c] = 0
            return False

        forbidden = False

        if Rules._is_overline(board, r, c, player):
            forbidden = True
        elif Rules._count_open_four(board, r, c, player) >= 2:
            forbidden = True
        elif Rules._count_real_open_three(board, r, c, player) >= 2:
            forbidden = True

        board[r, c] = 0
        return forbidden

    # 장목 판정
    @staticmethod
    def _is_overline(board, r, c, player):
        board_size = len(board)

        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    count += 1
                    nr    += dr * sign
                    nc    += dc * sign
            if count >= 6:
                return True
        return False

    # 열린 3 판정 : 실제 열린 3 개수 (거짓 쌍삼 제외)
    @staticmethod
    def _count_real_open_three(board, r, c, player):
        board_size = len(board)
        count      = 0

        for dr, dc in Rules.DIRECTIONS:
            if not Rules._is_open_three(
                    board, r, c, player, dr, dc, board_size):
                continue

            is_real = True
            ends    = Rules._get_open_ends(
                board, r, c, player, dr, dc, board_size)

            for er, ec in ends:
                if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0):
                    board[er, ec] = player
                    if Rules._is_overline(board, er, ec, player):
                        is_real = False
                    elif Rules._count_open_four(
                            board, er, ec, player) >= 2:
                        is_real = False
                    board[er, ec] = 0

            if is_real:
                count += 1

        return count

    @staticmethod
    def _get_open_ends(board, r, c, player, dr, dc, board_size):
        # 열린 3의 양 끝 좌표 반환
        ends = []
        for sign in [1, -1]:
            steps  = 0
            nr, nc = r + dr * sign, c + dc * sign
            while (0 <= nr < board_size and
                   0 <= nc < board_size and
                   board[nr, nc] == player):
                steps += 1
                nr    += dr * sign
                nc    += dc * sign
            ends.append((
                r + sign * (steps + 1) * dr,
                c + sign * (steps + 1) * dc))
        return ends

    @staticmethod
    def _is_open_three(board, r, c, player, dr, dc, board_size):
        # 한 방향 열린 3 판정
        for check in [
            Rules._check_open3_normal,
            Rules._check_open3_gap1,
            Rules._check_open3_gap2
        ]:
            if check(board, r, c, player, dr, dc, board_size):
                return True
        return False

    @staticmethod
    def _check_open3_normal(board, r, c, player, dr, dc, board_size):
        stones = 1
        ends   = []

        for sign in [1, -1]:
            steps  = 0
            nr, nc = r + dr * sign, c + dc * sign
            while (0 <= nr < board_size and
                   0 <= nc < board_size and
                   board[nr, nc] == player):
                stones += 1
                steps  += 1
                nr     += dr * sign
                nc     += dc * sign
            ends.append((
                r + sign * (steps + 1) * dr,
                c + sign * (steps + 1) * dc))

        if stones != 3:
            return False

        open_ends = sum(
            1 for er, ec in ends
            if (0 <= er < board_size and
                0 <= ec < board_size and
                board[er, ec] == 0))

        return open_ends == 2

    @staticmethod
    def _check_open3_gap1(board, r, c, player, dr, dc, board_size):
        seq = []
        for i in range(-3, 4):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                seq.append(int(board[nr, nc]))
            else:
                seq.append(-1)

        p   = player
        pat = [0, p, 0, p, p, 0]

        for start in range(len(seq) - len(pat) + 1):
            if seq[start:start + len(pat)] == pat:
                return True
        return False

    @staticmethod
    def _check_open3_gap2(board, r, c, player, dr, dc, board_size):
        seq = []
        for i in range(-3, 4):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                seq.append(int(board[nr, nc]))
            else:
                seq.append(-1)

        p   = player
        pat = [0, p, p, 0, p, 0]

        for start in range(len(seq) - len(pat) + 1):
            if seq[start:start + len(pat)] == pat:
                return True
        return False

    # 열린 4 판정
    @staticmethod
    def _count_open_four(board, r, c, player):
        # 열린 4 개수 반환
        board_size = len(board)
        count      = 0

        for dr, dc in Rules.DIRECTIONS:
            stones = 1
            ends   = []

            for sign in [1, -1]:
                steps  = 0
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    stones += 1
                    steps  += 1
                    nr     += dr * sign
                    nc     += dc * sign
                ends.append((
                    r + sign * (steps + 1) * dr,
                    c + sign * (steps + 1) * dc))

            if stones == 4:
                open_ends = sum(
                    1 for er, ec in ends
                    if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0))
                if open_ends >= 1:
                    count += 1

        return count