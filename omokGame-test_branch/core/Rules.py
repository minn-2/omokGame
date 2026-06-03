import numpy as np
from typing import List, Tuple, Optional

# 상수
EMPTY  = 0
BLACK  = 1   # 흑 (금수 적용)
WHITE  = 2   # 백

DIRECTIONS: List[Tuple[int, int]] = [
    (0, 1),   # →  가로
    (1, 0),   # ↓  세로
    (1, 1),   # ↘  대각선
    (1, -1),  # ↙  역대각선
]

# 내부 유틸
def _in_board(size: int, r: int, c: int) -> bool:
    return 0 <= r < size and 0 <= c < size

def _count_consecutive(board: np.ndarray, r: int, c: int,
                        dr: int, dc: int, player: int) -> int:
    size = board.shape[0]
    count = 1
    for sign in (1, -1):
        nr, nc = r + dr * sign, c + dc * sign
        while _in_board(size, nr, nc) and board[nr, nc] == player:
            count += 1
            nr += dr * sign
            nc += dc * sign
    return count

# 승리 판정
def check_win(board: np.ndarray, r: int, c: int) -> bool:
    player = board[r, c]
    if player == EMPTY:
        return False

    for dr, dc in DIRECTIONS:
        cnt = _count_consecutive(board, r, c, dr, dc, player)
        if player == BLACK and cnt == 5:
            return True
        if player == WHITE and cnt >= 5:
            return True
    return False

# 장목 (Overline)
def is_overline(board: np.ndarray, r: int, c: int, player: int) -> bool:
    for dr, dc in DIRECTIONS:
        if _count_consecutive(board, r, c, dr, dc, player) >= 6:
            return True
    return False

# 4 카운트 (Double Four 판정용)
def _direction_line(board: np.ndarray, r: int, c: int,
                    dr: int, dc: int, half: int = 4) -> List:
    size = board.shape[0]
    result = []
    for i in range(-half, half + 1):
        nr, nc = r + dr * i, c + dc * i
        if _in_board(size, nr, nc):
            result.append((nr, nc, int(board[nr, nc])))
        else:
            result.append((nr, nc, -1))  # 벽
    return result

def _has_four_in_direction(board: np.ndarray, r: int, c: int,
                            dr: int, dc: int, player: int) -> bool:
    size = board.shape[0]
    opponent = WHITE if player == BLACK else BLACK

    # (r,c) 기준 ±8 칸의 좌표 수집
    line = []
    for i in range(-8, 9):
        nr, nc = r + dr * i, c + dc * i
        if _in_board(size, nr, nc):
            line.append((i, nr, nc, int(board[nr, nc])))
        else:
            line.append((i, nr, nc, -1))  # 벽

    found = False
    n = len(line)
    for start in range(n - 4):
        window = line[start:start + 5]
        # (r,c) 가 이 윈도우에 포함되는지
        if not any(item[0] == 0 for item in window):
            continue
        vals = [item[3] for item in window]
        # 벽이나 상대돌이 있으면 스킵
        if -1 in vals or opponent in vals:
            continue
        # player 돌 4개 + 빈칸 1개
        if vals.count(player) == 4 and vals.count(EMPTY) == 1:
            # 빈칸 위치 찾기
            empty_idx = vals.index(EMPTY)
            _, er, ec, _ = window[empty_idx]
            # 완성수를 놓았을 때 장목이면 안 됨
            board[er, ec] = player
            overline = is_overline(board, er, ec, player)
            board[er, ec] = EMPTY
            if not overline:
                found = True
                break
    return found

def count_fours(board: np.ndarray, r: int, c: int, player: int) -> int:
    cnt = 0
    for dr, dc in DIRECTIONS:
        if _has_four_in_direction(board, r, c, dr, dc, player):
            cnt += 1
    return cnt

# 열린 4 (Open Four) – 삼삼 판정용 내부 함수
def _is_open_four_in_direction(board: np.ndarray, r: int, c: int,
                                dr: int, dc: int, player: int) -> bool:
    size = board.shape[0]
    opponent = WHITE if player == BLACK else BLACK

    # 연속 길이와 양 끝 상태 파악
    # (r,c) 포함 연속 구간의 양 끝 좌표를 찾음
    left_r, left_c = r, c
    right_r, right_c = r, c

    # 왼쪽(sign=-1) 방향으로 연속 확장
    nr, nc = r - dr, c - dc
    while _in_board(size, nr, nc) and board[nr, nc] == player:
        left_r, left_c = nr, nc
        nr -= dr
        nc -= dc

    # 오른쪽(sign=+1) 방향으로 연속 확장
    nr, nc = r + dr, c + dc
    while _in_board(size, nr, nc) and board[nr, nc] == player:
        right_r, right_c = nr, nc
        nr += dr
        nc += dc

    if dr != 0:
        length = abs(right_r - left_r) // abs(dr) + 1
    else:
        length = abs(right_c - left_c) // abs(dc) + 1

    if length != 4:
        return False

    # 양 끝 바깥이 빈칸인지 확인
    before_r, before_c = left_r - dr, left_c - dc
    after_r,  after_c  = right_r + dr, right_c + dc

    before_empty = (_in_board(size, before_r, before_c)
                    and board[before_r, before_c] == EMPTY)
    after_empty  = (_in_board(size, after_r, after_c)
                    and board[after_r, after_c] == EMPTY)

    return before_empty and after_empty

def _has_open_four_in_direction(board, r, c, dr, dc, player):
    return _is_open_four_in_direction(board, r, c, dr, dc, player)

# 열린 3 (Open Three) – 삼삼 판정용
def _has_open_three_in_direction(board: np.ndarray, r: int, c: int,
                                  dr: int, dc: int, player: int,
                                  _depth: int = 0) -> bool:
    size = board.shape[0]
    opponent = WHITE if player == BLACK else BLACK

    # ±7 칸 라인 수집 (gap이 있는 패턴 커버)
    line = []
    for i in range(-7, 8):
        nr, nc = r + dr * i, c + dc * i
        if _in_board(size, nr, nc):
            line.append((i, nr, nc, int(board[nr, nc])))
        else:
            line.append((i, nr, nc, -1))  # 벽

    # center는 line[7] (offset=0)
    n = len(line)

    def val_at(offset):
        idx = offset + 7
        if 0 <= idx < n:
            return line[idx][3]
        return -1

    def coord_at(offset):
        idx = offset + 7
        if 0 <= idx < n:
            return line[idx][1], line[idx][2]
        return None

    def try_completion(empty_offset: int) -> bool:
        coord = coord_at(empty_offset)
        if coord is None:
            return False
        er, ec = coord
        if board[er, ec] != EMPTY:
            return False

        board[er, ec] = player
        # 1) 이 방향으로 열린 4인지
        open4 = _is_open_four_in_direction(board, er, ec, dr, dc, player)
        if not open4:
            board[er, ec] = EMPTY
            return False
        # 2) 완성수 자체가 합법적인지 (금수 아닌지)
        #    _depth < 1 일 때만 재귀 33 체크 (렌주 공식 해석)
        if player == BLACK:
            if is_overline(board, er, ec, player):
                board[er, ec] = EMPTY
                return False
            if count_fours(board, er, ec, player) >= 2:
                board[er, ec] = EMPTY
                return False
            if _depth < 1:
                if count_open_threes(board, er, ec, player, _depth + 1) >= 2:
                    board[er, ec] = EMPTY
                    return False
        board[er, ec] = EMPTY
        return True

    for start in range(-5, 3):
        # 길이 5 윈도우
        w5 = [val_at(start + k) for k in range(5)]
        # center(offset=0) 가 윈도우 안에 있어야 함
        if not (start <= 0 <= start + 4):
            continue
        # 벽/상대돌 있으면 스킵
        if -1 in w5 or opponent in w5:
            pass  # 뒤에서 개별 처리
        else:
            # player 3개 + EMPTY 2개: 2개 빈칸 중 하나가 완성수
            if w5.count(player) == 3 and w5.count(EMPTY) == 2:
                empty_positions = [start + k for k in range(5) if w5[k] == EMPTY]
                # (r,c) 자신(offset=0)은 이미 player 돌이므로 빈칸이 아님 → 완성수 후보만
                for ep in empty_positions:
                    if try_completion(ep):
                        return True

    # gap 패턴: 길이 6 윈도우 (●●○●, ●○●● 등)
    for start in range(-5, 2):
        if not (start <= 0 <= start + 5):
            continue
        w6 = [val_at(start + k) for k in range(6)]
        # 양 끝이 EMPTY여야 열린 패턴
        if w6[0] != EMPTY or w6[5] != EMPTY:
            continue
        # 내부 4칸
        inner = w6[1:5]
        if -1 in inner or opponent in inner:
            continue
        if inner.count(player) == 3 and inner.count(EMPTY) == 1:
            gap_local = inner.index(EMPTY)  # 0~3
            gap_offset = start + 1 + gap_local
            if gap_offset == 0:
                continue  # (r,c) 자신은 이미 player
            if try_completion(gap_offset):
                return True

    return False

def count_open_threes(board: np.ndarray, r: int, c: int,
                      player: int, _depth: int = 0) -> int:
    cnt = 0
    for dr, dc in DIRECTIONS:
        if _has_open_three_in_direction(board, r, c, dr, dc, player, _depth):
            cnt += 1
    return cnt

# 금수 판정 (메인 API)
def is_forbidden(board: np.ndarray, r: int, c: int,
                 player: int = BLACK, _depth: int = 0) -> bool:
    if player != BLACK:
        return False
    if board[r, c] != EMPTY:
        return True  # 이미 돌이 있음

    board[r, c] = player

    # 1) 5목 완성 → 승리수, 금수 아님
    if check_win(board, r, c):
        board[r, c] = EMPTY
        return False

    # 2) 장목
    if is_overline(board, r, c, player):
        board[r, c] = EMPTY
        return True

    # 3) 44 (사사)
    if count_fours(board, r, c, player) >= 2:
        board[r, c] = EMPTY
        return True

    # 4) 33 (삼삼)
    if _depth < 1 and count_open_threes(board, r, c, player, _depth) >= 2:
        board[r, c] = EMPTY
        return True

    board[r, c] = EMPTY
    return False

def get_forbidden_reason(board: np.ndarray, r: int, c: int,
                          player: int = BLACK) -> Optional[str]:
    if player != BLACK:
        return None
    if board[r, c] != EMPTY:
        return "already_occupied"

    board[r, c] = player

    if check_win(board, r, c):
        board[r, c] = EMPTY
        return None

    if is_overline(board, r, c, player):
        board[r, c] = EMPTY
        return "overline"

    if count_fours(board, r, c, player) >= 2:
        board[r, c] = EMPTY
        return "double_four"

    if count_open_threes(board, r, c, player) >= 2:
        board[r, c] = EMPTY
        return "double_three"

    board[r, c] = EMPTY
    return None

# 보드 전체 금수 위치 탐색
def get_all_forbidden(board: np.ndarray, player: int = BLACK) -> List[Tuple[int, int]]:
    size = board.shape[0]
    result = []
    for r in range(size):
        for c in range(size):
            if board[r, c] == EMPTY and is_forbidden(board, r, c, player):
                result.append((r, c))
    return result

# 패턴 카운트 유틸 (기존 호환)
def check_patterns(board: np.ndarray, player: int, length: int) -> int:
    size = board.shape[0]
    count = 0
    seen = set()

    for r in range(size):
        for c in range(size):
            if board[r, c] != player:
                continue
            for dr, dc in DIRECTIONS:
                # 이 방향으로의 시작점만 카운트 (prev 방향에 같은 player 없을 때)
                prev_r, prev_c = r - dr, c - dc
                if (_in_board(size, prev_r, prev_c)
                        and board[prev_r, prev_c] == player):
                    continue
                match = True
                coords = [(r, c)]
                for i in range(1, length):
                    nr, nc = r + dr * i, c + dc * i
                    if not (_in_board(size, nr, nc)
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

# Rules 클래스 (기존 인터페이스 호환 래퍼)
class Rules:
    DIRECTIONS = DIRECTIONS

    @staticmethod
    def check_win(board: np.ndarray, r: int, c: int) -> bool:
        return check_win(board, r, c)

    @staticmethod
    def is_forbidden(board: np.ndarray, r: int, c: int,
                     player: int = BLACK, _depth: int = 0) -> bool:
        return is_forbidden(board, r, c, player, _depth)

    @staticmethod
    def get_forbidden_reason(board: np.ndarray, r: int, c: int,
                              player: int = BLACK) -> Optional[str]:
        return get_forbidden_reason(board, r, c, player)

    @staticmethod
    def get_all_forbidden(board: np.ndarray,
                          player: int = BLACK) -> List[Tuple[int, int]]:
        return get_all_forbidden(board, player)

    @staticmethod
    def check_patterns(board: np.ndarray, player: int, length: int) -> int:
        return check_patterns(board, player, length)

    @staticmethod
    def is_overline(board: np.ndarray, r: int, c: int, player: int) -> bool:
        return is_overline(board, r, c, player)

    @staticmethod
    def count_fours(board: np.ndarray, r: int, c: int, player: int) -> int:
        return count_fours(board, r, c, player)

    @staticmethod
    def count_open_threes(board: np.ndarray, r: int, c: int,
                          player: int, _depth: int = 0) -> int:
        return count_open_threes(board, r, c, player, _depth)

# 테스트
def _make_board(size=15):
    return np.zeros((size, size), dtype=int)

def _place(board, moves):
    for r, c, p in moves:
        board[r, c] = p

def _test_all():
    print("=" * 60)
    print("렌주룰 테스트 시작")
    print("=" * 60)
    passed = 0
    failed = 0

    def check(name, result, expected):
        nonlocal passed, failed
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  [{status}] {name}: got {result}, expected {expected}")
            failed += 1
        else:
            print(f"  [{status}] {name}")
            passed += 1

    # 1. 기본 승리 판정
    print("\n[1] 승리 판정")
    b = _make_board()
    for c in range(5):
        b[7, c] = BLACK
    check("흑 5목 가로 승리", check_win(b, 7, 4), True)

    b = _make_board()
    for c in range(6):
        b[7, c] = BLACK
    check("흑 장목(6목) 승리 아님", check_win(b, 7, 5), False)

    b = _make_board()
    for c in range(6):
        b[7, c] = WHITE
    check("백 장목(6목) 승리", check_win(b, 7, 5), True)

    # 2. 장목 금수
    print("\n[2] 장목 금수")
    b = _make_board()
    # ●●●●●● 흑 6목 시도
    for c in range(5):
        b[7, c] = BLACK
    check("흑 장목 금수", is_forbidden(b, 7, 5, BLACK), True)
    check("장목 이유 확인", get_forbidden_reason(b, 7, 5, BLACK), "overline")

    # 3. 사사(44) 금수
    print("\n[3] 사사(44) 금수")
    # 가로 4: _●●●_ + 세로 4: _●●●_ 교차
    b = _make_board()
    # 가로 방향 3개 (7행, 3~5열) → 7행 6열에 두면 가로 4
    b[7, 3] = BLACK
    b[7, 4] = BLACK
    b[7, 5] = BLACK
    # 세로 방향 3개 (4~6행, 6열) → 7행 6열에 두면 세로 4
    b[4, 6] = BLACK
    b[5, 6] = BLACK
    b[6, 6] = BLACK
    check("사사 금수", is_forbidden(b, 7, 6, BLACK), True)
    check("사사 이유 확인", get_forbidden_reason(b, 7, 6, BLACK), "double_four")

    # 4. 삼삼(33) 금수
    print("\n[4] 삼삼(33) 금수")
    # 가로 열린 3: _●●_ 에 한 칸 더 두면 열린 4
    # 세로 열린 3: 같은 교차점
    b = _make_board()
    # 가로: (7,3),(7,4) → (7,5)에 두면 _●●●_ (열린 3이 완성되어 열린 4가 가능)
    # 실제로 (7,5)에서 열린 3이 2개 교차하도록 설정
    # 가로: (7,4),(7,6) 사이 (7,5)
    b[7, 4] = BLACK
    b[7, 6] = BLACK
    # 세로: (5,5),(6,5) 사이 (7,5) → (7,5)에 두면 세로 3
    b[5, 5] = BLACK
    b[6, 5] = BLACK
    check("삼삼 금수", is_forbidden(b, 7, 5, BLACK), True)
    check("삼삼 이유 확인", get_forbidden_reason(b, 7, 5, BLACK), "double_three")

    # 5. 5목은 금수 아님 (승리수)
    print("\n[5] 5목 완성은 금수 아님")
    b = _make_board()
    for c in range(4):
        b[7, c] = BLACK
    check("흑 5목 완성 (금수 아님)", is_forbidden(b, 7, 4, BLACK), False)

    # 6. 백은 금수 없음
    print("\n[6] 백 금수 없음")
    b = _make_board()
    for c in range(5):
        b[7, c] = WHITE
    check("백 6목 금수 없음", is_forbidden(b, 7, 5, WHITE), False)

    # 7. 닫힌 4 는 사사에 포함
    print("\n[7] 닫힌 4 포함 사사")
    b = _make_board()
    # 가로 닫힌 4: ●●●_● → 0열이 막힘
    b[7, 0] = WHITE  # 막는 돌
    b[7, 1] = BLACK
    b[7, 2] = BLACK
    b[7, 3] = BLACK
    # 빈칸 7,4 에 두면 가로 4 완성 (7,5가 빈칸이어도 됨)
    # 세로 3개도 추가
    b[4, 4] = BLACK
    b[5, 4] = BLACK
    b[6, 4] = BLACK
    check("닫힌 4 포함 사사 금수", is_forbidden(b, 7, 4, BLACK), True)

    # 8. 사사여도 5목 완성이면 승리수
    print("\n[8] 사사 완성 지점이 동시에 5목이면 승리")
    b = _make_board()
    # 가로 4개: (7,3)(7,4)(7,6)(7,7) → (7,5)에 두면 가로 5목
    b[7, 3] = BLACK; b[7, 4] = BLACK; b[7, 6] = BLACK; b[7, 7] = BLACK
    # 세로 4개: (3,5)(4,5)(5,5)(6,5) → (7,5)에 두면 세로 5목
    b[3, 5] = BLACK; b[4, 5] = BLACK; b[5, 5] = BLACK; b[6, 5] = BLACK
    # (7,5) 에 두면 가로·세로 모두 5목 완성 (사사이지만 승리수) → 금수 아님
    check("사사지만 5목 완성 → 금수 아님", is_forbidden(b, 7, 5, BLACK), False)

    # 결과 요약
    print("\n" + "=" * 60)
    print(f"결과: {passed}개 통과 / {failed}개 실패 / 전체 {passed+failed}개")
    print("=" * 60)

if __name__ == "__main__":
    _test_all()