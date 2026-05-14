import numpy as np

class Rules:
    # 4방향 (양방향으로 8방향 커버)
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)] # 가로, 세로, 대각선

    # 승리 판정
    @staticmethod
    def check_win(board, r, c):
        player     = board[r, c] # 현재 위치의 플레이어
        board_size = len(board) # 보드 크기
        
        # 4방향 검사
        for dr, dc in Rules.DIRECTIONS:
            count = 1 # 현재 돌 포함
            
            # 양방향 탐색
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                
                # 같은 돌이 계속되면 개수 증가
                while (0 <= nr < board_size and 
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    count += 1
                    nr    += dr * sign
                    nc    += dc * sign

            if player == 1 and count == 5: # 흑돌 : 정확히 5목만 승리
                return True
            if player == 2 and count >= 5: # 백돌 : 5목 이상 승리
                return True

        return False
    
    # 금수 판정 (흑돌 전용)
    @staticmethod
    def is_forbidden(board, r, c, player):
        
        # 백돌은 금수 없음
        if player != 1:
            return False

        board[r, c] = player # 임시로 돌 놓기

        # 5목이면 금수 해제
        if Rules.check_win(board, r, c):
            board[r, c] = 0
            return False

        forbidden = False

        if Rules._is_overline(board, r, c, player): # 장목(6목 이상)
            forbidden = True
        elif Rules._count_open_four(board, r, c, player) >= 2: # 44 금수
            forbidden = True
        elif Rules._count_real_open_three(board, r, c, player) >= 2: # 33 금수
            forbidden = True

        board[r, c] = 0 # 원상복구
        return forbidden

    # 장목(6목 이상) 판정
    @staticmethod
    def _is_overline(board, r, c, player):
        board_size = len(board)

        for dr, dc in Rules.DIRECTIONS: # 4방향 검사
            count = 1

            # 양방향 탐색
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (0 <= nr < board_size and
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    count += 1
                    nr    += dr * sign
                    nc    += dc * sign

            # 6목 이상이면 장목
            if count >= 6:
                return True
        return False

    # 열린 3 판정 : 실제 열린 3 개수 (거짓 쌍삼 제외)
    @staticmethod
    def _count_real_open_three(board, r, c, player):
        board_size = len(board)
        count      = 0

        # 4방향 검사
        for dr, dc in Rules.DIRECTIONS:
            # 열린3이 아니면 넘어가기
            if not Rules._is_open_three(
                    board, r, c, player, dr, dc, board_size):
                continue

            # 진짜 열린3 여부
            is_real = True
            # 열린 양 끝 좌표 구하기
            ends    = Rules._get_open_ends(
                board, r, c, player, dr, dc, board_size)

            # 양 끝 검사
            for er, ec in ends:
                # 빈칸이면 가상 착수
                if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0):
                    board[er, ec] = player
                            
                    # 장목이면 가짜 열린3
                    if Rules._is_overline(board, er, ec, player):
                        is_real = False

                    # 44 발생 시 가짜 열린3
                    elif Rules._count_open_four(
                            board, er, ec, player) >= 2:
                        is_real = False
                    board[er, ec] = 0 # 원상복구

            # 진짜 열린3이면 개수 증가
            if is_real:
                count += 1

        return count

    @staticmethod
    def _get_open_ends(board, r, c, player, dr, dc, board_size):
        # 열린 양 끝 좌표 반환
        ends = []

        # 양방향 검사
        for sign in [1, -1]:
            steps  = 0
            nr, nc = r + dr * sign, c + dc * sign

            # 연속된 돌 개수 확인
            while (0 <= nr < board_size and
                   0 <= nc < board_size and
                   board[nr, nc] == player):
                steps += 1
                nr    += dr * sign
                nc    += dc * sign

            # 마지막 돌 다음 칸 저장
            ends.append((
                r + sign * (steps + 1) * dr,
                c + sign * (steps + 1) * dc))
        return ends

    @staticmethod
    def _is_open_three(board, r, c, player, dr, dc, board_size):
        # 한 방향 열린 3 판정
        for check in [
            Rules._check_open3_normal, # 기본 열린3
            Rules._check_open3_gap1, # 한 칸 띄운 열린3
            Rules._check_open3_gap2 # 다른 형태 열린3
        ]:
            if check(board, r, c, player, dr, dc, board_size): # 하나라도 만족하면 열린3
                return True
        return False

    @staticmethod
    def _check_open3_normal(board, r, c, player, dr, dc, board_size):
        # 기본 열린3 검사
        # 예: ○●●●○
        
        stones = 1
        ends   = []

        # 양방향 탐색
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

        # 정확히 3개인지 확인
        if stones != 3:
            return False

        # 열린 양 끝 개수 계산
        open_ends = sum(
            1 for er, ec in ends
            if (0 <= er < board_size and
                0 <= ec < board_size and
                board[er, ec] == 0)
        )

        # 양쪽 모두 열려 있어야 열린3
        return open_ends == 2

    # 열린3 검사
    # 예: ○●○●●○
    @staticmethod
    def _check_open3_gap1(board, r, c, player, dr, dc, board_size):
        seq = []
        
        # 주변 7칸 확인
        for i in range(-3, 4):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                seq.append(int(board[nr, nc]))
            else:
                seq.append(-1)

        p   = player
        pat = [0, p, 0, p, p, 0] # 패턴

        # 패턴 검사
        for start in range(len(seq) - len(pat) + 1):
            if seq[start:start + len(pat)] == pat:
                return True
        return False

    # 열린3 검사
    # 예: ○●●○●○
    @staticmethod
    def _check_open3_gap2(board, r, c, player, dr, dc, board_size):
        seq = []

        # 주변 7칸 확인
        for i in range(-3, 4):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                seq.append(int(board[nr, nc]))
            else:
                seq.append(-1)

        p   = player
        pat = [0, p, p, 0, p, 0] # 패턴

        # 패턴 검사
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

        # 4방향 검사
        for dr, dc in Rules.DIRECTIONS:
            stones = 1
            ends   = []

            # 양방향 탐색
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

            # 연속된 4개인지 검사
            if stones == 4:

                # 열린 끝 개수 계산
                open_ends = sum(
                    1 for er, ec in ends
                    if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0)
                )

                # 한쪽 이상 열려 있으면 열린4
                if open_ends >= 1:
                    count += 1

        return count
    # 패턴 검사 (AI 학습 보상 계산용)
    @staticmethod
    def check_patterns(board, player, length):
        board_size = len(board)
        count = 0
        
        # 가로, 세로, 대각선 모든 위치에서 해당 길이의 연속된 돌이 있는지 검사
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == player:
                    for dr, dc in Rules.DIRECTIONS:
                        # 한 방향으로 탐색
                        match = True
                        for i in range(1, length):
                            nr, nc = r + dr * i, c + dc * i
                            if not (0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player):
                                match = False
                                break
                        
                        if match:
                            # 양 끝이 막혀있는지 등 세부 조건 없이 단순 길이만 체크
                            count += 1
        return count
