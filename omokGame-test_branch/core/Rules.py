import numpy as np

class Rules:

    DIRECTIONS = [
        (0, 1),   # 가로
        (1, 0),   # 세로
        (1, 1),   # 우하 대각선
        (1, -1)   # 좌하 대각선
    ]

    @staticmethod
    def check_win(board, r, c):
        player     = board[r, c]
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
            
            if player == 1 and count == 5:  # 흑은 정확히 5목만 승리 (6목 이상은 장목 금수)
                return True
            if player == 2 and count >= 5:  # 백은 5목 이상이면 모두 승리
                return True
        return False

    @staticmethod
    def is_forbidden(board, r, c, player):
        if player != 1:  # 백은 금수가 없음
            return False
        if board[r, c] != 0:  # 이미 돌이 놓인 곳은 둘 수 없음
            return True

        # 1. 가상 착수
        board[r, c] = player

        # 렌주룰 핵심: 금수 자리라도 '동시에 5목'이 완성되면 금수가 아니라 승리입니다.
        if Rules.check_win(board, r, c):
            board[r, c] = 0
            return False

        # 2. 금수 조건 순차적 판정
        forbidden = False
        
        if Rules._is_overline(board, r, c, player):
            forbidden = True
        elif Rules._count_legal_fours(board, r, c, player) >= 2:   # 사사(4-4) 금수
            forbidden = True
        elif Rules._count_legal_threes(board, r, c, player) >= 2:  # 삼삼(3-3) 금수
            forbidden = True

        # 3. 보드 원상복구
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
            if count >= 6:
                return True
        return False

    @staticmethod
    def _count_legal_fours(board, r, c, player):
        board_size = len(board)
        four_count = 0

        for dr, dc in Rules.DIRECTIONS:
            # 이 방향상에서 흑이 한 번 더 두면 5목이 되는 빈칸(공격점)의 좌표를 찾습니다.
            found_four_for_this_dir = False
            
            # 현재 착수점을 중심으로 양방향 총 5칸 반경 탐색
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                
                # 보드 범위를 벗어나거나 빈칸이 아니면 패스
                if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr, nc] != 0:
                    continue
                
                # 그 빈칸에 가상으로 흑돌을 놓아봅니다.
                board[nr, nc] = player
                # 흑이 정확히 5목을 완성하는 자리라면, 현재 만들어진 선은 유효한 '4'입니다.
                if Rules.check_win(board, nr, nc):
                    found_four_for_this_dir = True
                board[nr, nc] = 0 # 원상복구
                
                if found_four_for_this_dir:
                    break
            
            if found_four_for_this_dir:
                four_count += 1

        return four_count

    @staticmethod
    def _count_legal_threes(board, r, c, player):
        board_size = len(board)
        three_count = 0

        for dr, dc in Rules.DIRECTIONS:
            is_open_three = False
            
            # 현재 착수점 중심 반경 탐색하여 다음 '4'를 만들 수 있는 빈칸을 찾음
            for i in range(-4, 5):
                if i == 0: continue
                nr, nc = r + i * dr, c + i * dc
                
                if not (0 <= nr < board_size and 0 <= nc < board_size) or board[nr, nc] != 0:
                    continue
                
                # 가상으로 다음 수를 놓아봅니다.
                board[nr, nc] = player
                
                # 다음 수를 놓았을 때 '열린 4'가 되는지 확인합니다.
                # 열린 4 조건: 6목(장목)이 아니면서, 다음 5목을 완성할 수 있는 유효한 공격점이 '양쪽(2개)' 존재해야 함
                if not Rules._is_overline(board, nr, nc, player):
                    win_spots = 0
                    for j in range(-4, 5):
                        nnr, nnc = nr + j * dr, nc + j * dc
                        if not (0 <= nnr < board_size and 0 <= nnc < board_size) or board[nnr, nnc] != 0:
                            continue
                        
                        board[nnr, nnc] = player
                        if Rules.check_win(board, nnr, nnc):
                            win_spots += 1
                        board[nnr, nnc] = 0
                    
                    # 양방향으로 승리할 수 있는 '열린 4' 조건 만족 시
                    if win_spots >= 2:
                        # 거짓 3 판정: 3에서 4로 발전시키기 위해 두는 그 자리(nr, nc)가 
                        # 자체적으로 또 다른 금수 자리(예: 사사금수 등)라면 흑은 거기 둘 수 없으므로 이 3은 '가짜 3'입니다.
                        board[nr, nc] = 0 # 잠시 풀고 금수 체크
                        if not Rules.is_forbidden(board, nr, nc, player):
                            is_open_three = True
                        board[nr, nc] = player # 다시 원상복구
                        
                board[nr, nc] = 0
                if is_open_three:
                    break
                    
            if is_open_three:
                three_count += 1

        return three_count