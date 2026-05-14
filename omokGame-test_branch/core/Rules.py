import numpy as np

class Rules:
    # [설정] 오목 판정을 위한 4가지 방향 정의 (양방향 탐색 시 8방향 모두 커버)
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)] # 가로, 세로, 우하향 대각선, 우상향 대각선

    # [핵심 로직] 승리 판정 함수
    @staticmethod
    def check_win(board, r, c):
        player     = board[r, c] # 현재 착수한 플레이어 번호 (1:흑, 2:백)
        board_size = len(board) # 보드 크기 
        
        # 4가지 모든 방향에 대해 연속된 돌의 개수를 세기 시작
        for dr, dc in Rules.DIRECTIONS:
            count = 1 # 현재 놓은 돌을 포함하여 1부터 시작
            
            # 한 방향(sign=1)과 그 반대 방향(sign=-1)을 모두 확인하여 총 개수 합산
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                
                # 보드 범위를 벗어나지 않고, 같은 색의 돌이 연속되는 동안 계속 이동
                while (0 <= nr < board_size and 
                       0 <= nc < board_size and
                       board[nr, nc] == player):
                    count += 1
                    nr    += dr * sign
                    nc    += dc * sign

            # [렌주룰 적용] 흑돌은 반드시 정확히 5목이어야 승리 (6목 이상인 장목은 승리 인정 안 됨)
            if player == 1 and count == 5: 
                return True
            # [렌주룰 적용] 백돌은 5목 이상(장목 포함)이면 모두 승리 인정
            if player == 2 and count >= 5: 
                return True

        return False
    
    # [핵심 로직] 흑돌 전용 금수(착수 금지) 판정 함수
    @staticmethod
    def is_forbidden(board, r, c, player):
        
        # 렌주룰에서 금수는 오직 흑돌(1)에게만 적용됨 (백돌은 제약 없음)
        if player != 1:
            return False

        # 시뮬레이션을 위해 해당 자리에 임시로 돌을 놓음
        board[r, c] = player 

        # [예외 규정] 흑돌이 금수 자리에 놓았더라도 동시에 5목이 완성된다면 승리가 우선
        if Rules.check_win(board, r, c):
            board[r, c] = 0 # 원상복구
            return False

        forbidden = False

        # 1. 장목 금수 판정 (6목 이상 연속되는 경우)
        if Rules._is_overline(board, r, c, player): 
            forbidden = True
        # 2. 44 금수 판정 (착수 후 열린 4가 2개 이상 만들어지는 경우)
        elif Rules._count_open_four(board, r, c, player) >= 2: 
            forbidden = True
        # 3. 33 금수 판정 (착수 후 진짜 열린 3이 2개 이상 만들어지는 경우)
        elif Rules._count_real_open_three(board, r, c, player) >= 2: 
            forbidden = True

        board[r, c] = 0 # 시뮬레이션 종료 후 원상복구
        return forbidden

    # [보조 함수] 장목(6개 이상 연속)인지 확인
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

    # [보조 함수]
    @staticmethod
    def _count_real_open_three(board, r, c, player):
        board_size = len(board)
        count      = 0

        for dr, dc in Rules.DIRECTIONS:
            # 우선 해당 방향으로 열린 3의 형태가 만들어지는지 확인
            if not Rules._is_open_three(
                    board, r, c, player, dr, dc, board_size):
                continue

            is_real = True
            # 열린 3의 양 끝(비어있는 곳) 좌표를 가져옴
            ends    = Rules._get_open_ends(
                board, r, c, player, dr, dc, board_size)

            # 양 끝 빈칸에 돌을 놓았을 때 금수가 발생한다면 그 3은 '열린 3'이 아님
            for er, ec in ends:
                if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0):
                    board[er, ec] = player
                            
                    # 만약 그 자리가 장목이 되거나 44 자리가 된다면 '가짜' 열린 3으로 판정
                    if Rules._is_overline(board, er, ec, player):
                        is_real = False

                    elif Rules._count_open_four(
                            board, er, ec, player) >= 2:
                        is_real = False
                    board[er, ec] = 0 

            # 모든 검증을 통과한 진짜 열린 3만 카운트
            if is_real:
                count += 1

        return count

    # [보조 함수] 열린 형태의 양 끝 빈 좌표 반환
    @staticmethod
    def _get_open_ends(board, r, c, player, dr, dc, board_size):
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

            # 연속된 돌이 끝나는 바로 다음 칸의 좌표 저장
            ends.append((
                r + sign * (steps + 1) * dr,
                c + sign * (steps + 1) * dc))
        return ends

    # [보조 함수] 한 방향에 대해 열린 3 패턴이 있는지 확인
    @staticmethod
    def _is_open_three(board, r, c, player, dr, dc, board_size):
        # 3가지 형태(붙어있는 3, 한 칸 띈 3 등) 중 하나라도 만족하면 True
        for check in [
            Rules._check_open3_normal, # ○●●●○ 형태
            Rules._check_open3_gap1,   # ○●○●●○ 형태
            Rules._check_open3_gap2    # ○●●○●○ 형태
        ]:
            if check(board, r, c, player, dr, dc, board_size): 
                return True
        return False

    # [패턴 매칭] 가장 기본적인 열린 3 (연속된 3개와 양 끝 공백)
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

        # 양 끝이 모두 보드 안쪽이며 비어있는(0) 상태여야 함
        open_ends = sum(
            1 for er, ec in ends
            if (0 <= er < board_size and
                0 <= ec < board_size and
                board[er, ec] == 0)
        )

        return open_ends == 2

    # [패턴 매칭] 징검다리 형태 열린 3 (중간에 한 칸 빈 경우: ○●○●●○)
    @staticmethod
    def _check_open3_gap1(board, r, c, player, dr, dc, board_size):
        seq = []
        
        # 현재 위치를 중심으로 앞뒤 3칸씩(총 7칸)의 상태를 리스트화
        for i in range(-3, 4):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                seq.append(int(board[nr, nc]))
            else:
                seq.append(-1) # 보드 밖은 -1로 처리

        p   = player
        pat = [0, p, 0, p, p, 0] 

        # 생성된 리스트 안에 해당 패턴이 슬라이딩 윈도우 방식으로 존재하는지 
        for start in range(len(seq) - len(pat) + 1):
            if seq[start:start + len(pat)] == pat:
                return True
        return False

    # [패턴 매칭] 징검다리 형태 열린 3 (중간에 한 칸 빈 경우: ○●●○●○)
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

    # [보조 함수] 열린 4의 개수를 세는 함수
    @staticmethod
    def _count_open_four(board, r, c, player):
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

            # 연속된 4개가 만들어진 경우
            if stones == 4:
                # 렌주룰에서 44금수는 한쪽만 열린 4(사구)여도 해당
                open_ends = sum(
                    1 for er, ec in ends
                    if (0 <= er < board_size and
                        0 <= ec < board_size and
                        board[er, ec] == 0)
                )

                if open_ends >= 1:
                    count += 1

        return count

    # [AI 보상용] 보드 전체에서 특정 길이의 패턴이 몇 개 있는지 검사
    @staticmethod
    def check_patterns(board, player, length):
        board_size = len(board)
        count = 0
        
        # 전수 조사를 통해 모든 좌표와 방향 확인
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == player:
                    for dr, dc in Rules.DIRECTIONS:
                        match = True
                        for i in range(1, length):
                            nr, nc = r + dr * i, c + dc * i
                            # 해당 길이만큼 돌이 끊기지 않고 이어지는지 확인
                            if not (0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player):
                                match = False
                                break
                        
                        if match:
                            count += 1
        # 양방향으로 중복 카운트되므로 방향의 개수만큼 보정 필요
        return count
