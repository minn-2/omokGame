import numpy as np


class Rules:
    DIRECTIONS = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1)
    ]

    @staticmethod
    def check_win(board, r, c):
        player = board[r, c]
        board_size = len(board)
        if player == 0:
            return False

        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size
                       and 0 <= nc < board_size
                       and board[nr, nc] == player):
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            if player == 1 and count == 5:
                return True
            if player == 2 and count >= 5:
                return True
        return False

    @staticmethod
    def is_forbidden(board, r, c, player, check_only=False):
        """
        흑돌(player=1) 금수 판정.
        check_only=True : 삼삼 체크를 생략 (재귀 루프 방지용).
                          _count_legal_threes 내부에서만 True로 호출.
        """
        if player != 1:
            return False
        if board[r, c] != 0:
            return True

        board[r, c] = player

        # 5목 완성 → 금수 아님
        if Rules.check_win(board, r, c):
            board[r, c] = 0
            return False

        forbidden = False
        if Rules._is_overline(board, r, c, player):
            # 장목 (6목 이상)
            forbidden = True
        elif Rules._count_legal_fours(board, r, c, player) >= 2:
            # 사사 (열린 4가 2개 이상)
            forbidden = True
        elif not check_only and Rules._count_legal_threes(board, r, c, player) >= 2:
            # 삼삼 (열린 3이 2개 이상)
            forbidden = True

        board[r, c] = 0
        return forbidden

    @staticmethod
    def _is_overline(board, r, c, player):
        """장목: 착수점 기준 6목 이상."""
        board_size = len(board)
        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size
                       and 0 <= nc < board_size
                       and board[nr, nc] == player):
                    count += 1
                    nr += dr * sign
                    nc += dc * sign
            if count >= 6:
                return True
        return False

    @staticmethod
    def _count_legal_fours(board, r, c, player):
        """
        사사 판정: 방향별로 빈칸에 한 수 더 놓아 5목이 되는 방향 수를 센다.
        방향당 최대 1회 카운트 (found 시 break).
        """
        board_size = len(board)
        four_count = 0

        for dr, dc in Rules.DIRECTIONS:
            found = False
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < board_size
                        and 0 <= nc < board_size):
                    continue
                if board[nr, nc] != 0:
                    continue
                board[nr, nc] = player
                if Rules.check_win(board, nr, nc):
                    found = True
                board[nr, nc] = 0
                if found:
                    break
            if found:
                four_count += 1

        return four_count

    @staticmethod
    def _count_legal_threes(board, r, c, player):
        """
        삼삼 판정: 방향별로 '열린 3' 개수를 센다.

        열린 3 조건:
          1. 착수점 근처 빈칸(nr,nc)에 한 수 더 놓으면
          2. 장목이 되지 않으면서
          3. 그 상태에서 5목을 만드는 빈칸이 2개 이상 존재하고 (= 열린 4)
          4. 그 중간 상태(nr,nc)가 금수가 아닐 것
             (재귀 방지를 위해 삼삼 체크는 생략: check_only=True)
        """
        board_size = len(board)
        three_count = 0

        for dr, dc in Rules.DIRECTIONS:
            is_open_three = False

            for i in range(-4, 5):
                if i == 0:
                    continue
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < board_size
                        and 0 <= nc < board_size):
                    continue
                if board[nr, nc] != 0:
                    continue

                board[nr, nc] = player

                if not Rules._is_overline(board, nr, nc, player):
                    win_spots = 0
                    for j in range(-4, 5):
                        nnr, nnc = nr + j * dr, nc + j * dc
                        if not (0 <= nnr < board_size
                                and 0 <= nnc < board_size):
                            continue
                        if board[nnr, nnc] != 0:
                            continue
                        board[nnr, nnc] = player
                        if Rules.check_win(board, nnr, nnc):
                            win_spots += 1
                        board[nnr, nnc] = 0

                    if win_spots >= 2:
                        board[nr, nc] = 0
                        if not Rules.is_forbidden(board, nr, nc, player,
                                                  check_only=True):
                            is_open_three = True
                        board[nr, nc] = player

                board[nr, nc] = 0
                if is_open_three:
                    break

            if is_open_three:
                three_count += 1

        return three_count

    @staticmethod
    def check_patterns(board, player, length):
        """보드 전체에서 player 의 length 연속 패턴 수를 반환."""
        board_size = len(board)
        count = 0
        seen  = set()
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] != player:
                    continue
                for dr, dc in Rules.DIRECTIONS:
                    prev_r, prev_c = r - dr, c - dc
                    if (0 <= prev_r < board_size
                            and 0 <= prev_c < board_size
                            and board[prev_r, prev_c] == player):
                        continue
                    match  = True
                    coords = [(r, c)]
                    for i in range(1, length):
                        nr, nc = r + dr * i, c + dc * i
                        if not (0 <= nr < board_size
                                and 0 <= nc < board_size
                                and board[nr, nc] == player):
                            match = False
                            break
                        coords.append((nr, nc))
                    if match:
                        key = tuple(coords)
                        if key not in seen:
                            seen.add(key)
                            count += 1
        return count