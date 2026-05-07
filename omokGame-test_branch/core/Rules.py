class Rules:
    # 4방향 (양방향으로 8방향 커버)
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    @staticmethod
    def check_win(board, r, c):
        """
        8방향 탐색으로 5목 승리 판정
        방금 둔 돌 기점으로 정방향 + 역방향 탐색
        """
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
            if count >= 5:
                return True
        return False

    @staticmethod
    def check_patterns(board, player, length):
        """
        3목 / 4목 패턴 감지
        AI 중간 보상 계산에 사용
        """
        board_size = len(board)

        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == player:
                    for dr, dc in Rules.DIRECTIONS:
                        count = 1
                        nr, nc = r + dr, c + dc
                        while (0 <= nr < board_size and
                               0 <= nc < board_size and
                               board[nr, nc] == player):
                            count += 1
                            nr    += dr
                            nc    += dc
                        if count == length:
                            return True
        return False