import numpy as np

class Rules:
    
    DIRECTIONS = [
        (0, 1),   # 가로 (우측 방향)
        (1, 0),   # 세로 (하단 방향)
        (1, 1),   # 우하향 대각선 (↘)
        (1, -1)   # 우상향 대각선 (↗)
    ]

    # 33금수
    OPEN_THREE_PATTERNS = [
        [0, 1, 1, 1, 0],          # 1) 기본 열린 3       ( . ○ ○ ○ . )
        [0, 1, 1, 0, 1, 0],       # 2) 한 칸 띈 3 (우)   ( . ○ ○ . ○ . )
        [0, 1, 0, 1, 1, 0],       # 3) 한 칸 띈 3 (좌)   ( . ○ . ○ ○ . )
        [0, 1, 1, 0, 0, 1, 0],    # 4) 두 칸 띈 변형 3   ( . ○ ○ . . ○ . )
        [0, 1, 0, 1, 0, 1, 0],    # 5) 띄엄띄엄 형태의 3 ( . ○ . ○ . ○ . )
    ]

    # 44금수
    OPEN_FOUR_PATTERNS = [
        [0, 1, 1, 1, 1, 0],       # 1) 기본 열린 4
        [0, 1, 1, 1, 0, 1, 0],    # 2) 한 칸 끼운 4 
        [0, 1, 1, 0, 1, 1, 0],    # 3) 가운데 끼운 4
        [0, 1, 0, 1, 1, 1, 0],    # 4) 한 칸 끼운 4
    ]

    # 승리 판정 함수 : 특정 좌표(r, c)에 돌이 놓였을 때, 5목 이상이 완성되어 승리했는지 판정
    @staticmethod
    def check_win(board, r, c):
        player = board[r, c]  # 현재 착수한 플레이어 번호 확인 (1: 흑돌, 2: 백돌)
        board_size = len(board)

        # 정의된 4가지 모든 방향에 대해 연속된 돌의 개수를 누적 계산
        for dr, dc in Rules.DIRECTIONS:
            count = 1  # 방금 놓은 돌 자체를 개수에 포함 (기본값 1)

            # 한 방향(sign=1)과 그 정반대 방향(sign=-1)을 모두 뻗어가며 확인
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign

                # 바둑판 범위를 벗어나지 않고, 방금 놓은 돌과 같은 색인 동안 계속 전진
                while (
                    0 <= nr < board_size
                    and 0 <= nc < board_size
                    and board[nr, nc] == player
                ):
                    count += 1         # 연속된 돌 개수 증가
                    nr += dr * sign    # 같은 방향으로 한 칸 더 이동
                    nc += dc * sign

            # 1. 흑돌(1)은 반드시 '정확히 5목'일 때만 승리 인정 (6목 이상의 장목은 금수이므로 무효)
            if player == 1 and count == 5:
                return True
            # 2. 백돌(2)은 렌주룰 제약이 없으므로, 5목이든 6목 이상(장목)이든 전부 승리 인정
            if player == 2 and count >= 5:
                return True

        return False

    # 흑돌 전용 금수(착수 금지) 판정 함수 : 렌주룰에 따라 흑돌이 두면 안 되는 자리를 필터링 (장목, 44, 33 순으로 체크)
    @staticmethod
    def is_forbidden(board, r, c, player):
        # 렌주룰의 금수 규칙은 오직 '흑돌(1)'에게만 적용됨 (백돌은 제약 없이 통과)
        if player != 1:
            return False

        # 이미 돌이 놓여 있는 자리에는 당연히 착수 불가하므로 금수 처리
        if board[r, c] != 0:
            return True

        # 가상으로 착수 시뮬레이션을 하기 위해 해당 자리에 흑돌을 임시로 임베딩
        board[r, c] = player

        # 착수한 자리가 금수 조건(33, 44)에 해당하더라도, '동시에 5목이 완성'된다면 5목이 우선됨
        if Rules.check_win(board, r, c):
            board[r, c] = 0  # 시뮬레이션 복구 후 리턴
            return False

        forbidden = False

        # 장목 금수 판정 (돌이 연속으로 6개 이상 놓이게 되는지 확인)
        if Rules._is_overline(board, r, c, player):
            forbidden = True
            
        # 44 금수 판정 (착수 후 정상적인 '열린 4'가 동시에 2개 이상 생성되는지 확인)
        elif Rules._count_legal_open_fours(board, r, c, player) >= 2:
            forbidden = True
            
        # 33 금수 판정 (착수 후 정상적인 '열린 3'이 동시에 2개 이상 생성되는지 확인)
        elif Rules._count_legal_open_threes(board, r, c, player) >= 2:
            forbidden = True

        # 시뮬레이션이 끝났으므로 임시로 임베딩했던 바둑판 좌표를 다시 빈칸으로 복구
        board[r, c] = 0
        return forbidden

    # 장목(6목 이상 연속) 유무 검사
    @staticmethod
    def _is_overline(board, r, c, player):
        board_size = len(board)

        for dr, dc in Rules.DIRECTIONS:
            count = 1
            for sign in [1, -1]:
                nr, nc = r + dr * sign, c + dc * sign
                while (
                    0 <= nr < board_size
                    and 0 <= nc < board_size
                    and board[nr, nc] == player
                ):
                    count += 1
                    nr += dr * sign
                    nc += dc * sign

            # 한 방향이라도 연속된 돌의 누적 합산이 6개 이상이면 장목 확정
            if count >= 6:
                return True

        return False
   
    # 보드 전체를 전수조사하여 특정 길이의 단순 직선 패턴 개수 카운트
    @staticmethod
    def check_patterns(board, player, length):
        board_size = len(board)
        count = 0
        seen = set()  # 동일한 돌 묶음이 중복으로 카운트되는 것을 방지하기 위한 세트

        # 바둑판 모든 좌표를 탐색
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] != player:
                    continue

                for dr, dc in Rules.DIRECTIONS:
                    # 패턴의 시작점을 유일하게 잡기 위해, 현재 돌의 이전 칸이 같은 플레이어의 돌이면 스킵
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < board_size and 0 <= prev_c < board_size and board[prev_r, prev_c] == player:
                        continue

                    match = True
                    coords = [(r, c)]

                    # 지정된 길이만큼 한 방향으로 뻗어나가며 연속되는지 확인
                    for i in range(1, length):
                        nr, nc = r + dr * i, c + dc * i
                        if not (
                            0 <= nr < board_size
                            and 0 <= nc < board_size
                            and board[nr, nc] == player
                        ):
                            match = False
                            break
                        coords.append((nr, nc))

                    # 원하는 연속 길이를 충족했다면 중복 체크 후 카운트 반영
                    if match:
                        key = tuple(coords)
                        if key not in seen:
                            seen.add(key)
                            count += 1

        return count
   
    # 특정 착수 지점에 형성되는 '열린 3'의 개수를 산출
    @staticmethod
    def _count_legal_open_threes(board, r, c, player):
        return Rules._count_pattern_matches(board, r, c, player, Rules.OPEN_THREE_PATTERNS)

    # 특정 착수 지점에 형성되는 '열린 4'의 개수를 산출
    @staticmethod
    def _count_legal_open_fours(board, r, c, player):
        return Rules._count_pattern_matches(board, r, c, player, Rules.OPEN_FOUR_PATTERNS)

    # 슬라이딩 윈도우 기반 패턴 매칭 공통 알고리즘
    # - 특정 착수 지점(r, c)을 중심으로 1D 라인 데이터를 추출해 상단 패턴 배열과 비교 매칭
    @staticmethod
    def _count_pattern_matches(board, r, c, player, patterns):
        board_size = len(board)
        seen = set()  # 중복 카운트 방지 세트
        count = 0

        # 가로, 세로, 대각선 대칭축 전 방향 검사
        for dr, dc in Rules.DIRECTIONS:
            # 착수 지점을 중심으로 앞뒤 6칸씩 수평선 추출
            line, positions = Rules._get_line_with_pos(board, r, c, dr, dc, board_size, radius=6)

            # 비교하고자 하는 패턴 목록을 하나씩 대조
            for pat in patterns:
                plen = len(pat)
                # 슬라이딩 윈도우 기법으로 1D 라인 위를 한 칸씩 이동하며 패턴 비교
                for start in range(len(line) - plen + 1):
                    chunk = line[start:start + plen]
                    
                    # 오목판에서 자른 조각이 타겟 패턴과 일치하는 경우
                    if chunk == pat:
                        # 중복 인식을 차단하기 위해 매칭된 실제 오목판 좌표를 키값으로 캡처
                        pos_chunk = positions[start:start + plen]
                        key = (dr, dc, tuple(pos_chunk), tuple(chunk))
                        
                        if key not in seen:
                            seen.add(key)
                            count += 1

        return count

    # 타겟 좌표 중심의 1차원 라인 배열 데이터 및 좌표 맵 추출
    # - 바둑판 밖의 영역은 패턴 매칭 시 오작동하지 않도록 벽면 패딩값(-1) 처리
    @staticmethod
    def _get_line_with_pos(board, r, c, dr, dc, board_size, radius=6):
        line = []
        positions = []

        # -radius칸 앞부터 +radius칸 뒤까지 순차적으로 1차원 공간 확보
        for i in range(-radius, radius + 1):
            nr, nc = r + i * dr, c + i * dc
            positions.append((nr, nc)) # 좌표 매핑 기록
            
            # 유효한 바둑판 내부 영역인 경우 보드 상태값(0: 빈칸, 1: 흑, 2: 백) 기록
            if 0 <= nr < board_size and 0 <= nc < board_size:
                line.append(int(board[nr, nc]))
            # 바둑판 경계를 완전히 넘어선 외곽벽 부분은 예외값 처리 (-1)
            else:
                line.append(-1)

        return line, positions
