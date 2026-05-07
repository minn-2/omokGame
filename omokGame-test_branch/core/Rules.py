# ============================================================
# gomoku_rules.py  —  오목 규칙 모듈
# ============================================================
# 담당 역할
#   - 보드 위 특정 위치에서 승리 조건(5목) 판정
#   - 흑(1번 플레이어)의 금수 판정: 3-3, 4-4, 장목(6목 이상)
#   - 방향 벡터를 이용한 연속 돌 카운팅 유틸리티
# ============================================================

EMPTY = 0
BLACK = 1   # 플레이어 1 (흑)
WHITE = 2   # 플레이어 2 (백)

# 8방향 → 4쌍(수평, 수직, 대각선 2개)으로 묶어서 사용
DIRECTIONS = [
    (0, 1),   # 가로
    (1, 0),   # 세로
    (1, 1),   # 대각선 ↘
    (1, -1),  # 대각선 ↙
]


def count_in_direction(board, row, col, dr, dc, player):
    """
    (row, col)에서 (dr, dc) 방향과 반대 방향으로
    player의 연속된 돌 수를 반환한다.
    """
    size = len(board)
    count = 1  # 현재 위치 포함

    # 정방향
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        count += 1
        r += dr
        c += dc

    # 역방향
    r, c = row - dr, col - dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        count += 1
        r -= dr
        c -= dc

    return count


def check_win(board, row, col, player):
    """
    (row, col)에 player의 돌을 놓은 뒤 승리 여부를 반환한다.
    - 백(WHITE)은 5목 이상이면 승리.
    - 흑(BLACK)은 정확히 5목이어야 승리(장목 제외).
    """
    for dr, dc in DIRECTIONS:
        cnt = count_in_direction(board, row, col, dr, dc, player)
        if player == WHITE and cnt >= 5:
            return True
        if player == BLACK and cnt == 5:
            return True
    return False


# ── 금수 판정 (흑 전용) ──────────────────────────────────────

def _open_count(board, row, col, dr, dc, player):
    """
    한 방향+역방향의 연속 돌 수(cnt)와
    양 끝이 열려 있는지(open) 여부를 반환한다.
    반환: (연속 수, 양끝 열린 수)
    """
    size = len(board)
    cnt = 1
    open_ends = 0

    # 정방향
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        cnt += 1
        r += dr
        c += dc
    if 0 <= r < size and 0 <= c < size and board[r][c] == EMPTY:
        open_ends += 1

    # 역방향
    r, c = row - dr, col - dc
    while 0 <= r < size and 0 <= c < size and board[r][c] == player:
        cnt += 1
        r -= dr
        c -= dc
    if 0 <= r < size and 0 <= c < size and board[r][c] == EMPTY:
        open_ends += 1

    return cnt, open_ends


def is_forbidden(board, row, col):
    """
    흑(BLACK) 전용 금수 판정.
    금수 조건:
      1. 장목(overline): 6목 이상
      2. 4-4: 두 방향 이상에서 열린 4(또는 닫힌 4)가 동시에 완성
      3. 3-3: 두 방향 이상에서 열린 3이 동시에 완성
    반환: (is_forbidden: bool, reason: str)
    """
    # 임시로 돌을 놓은 상태라고 가정 (board[row][col]이 이미 BLACK)
    assert board[row][col] == BLACK

    fours = 0   # 4목 카운트
    threes = 0  # 열린 3목 카운트

    for dr, dc in DIRECTIONS:
        cnt = count_in_direction(board, row, col, dr, dc, BLACK)

        # 1. 장목
        if cnt >= 6:
            return True, "장목(6목 이상)"

        # 4목 여부
        if cnt == 4:
            _, open_ends = _open_count(board, row, col, dr, dc, BLACK)
            fours += 1  # 열린 4, 닫힌 4 모두 카운트

        # 열린 3목 여부
        if cnt == 3:
            _, open_ends = _open_count(board, row, col, dr, dc, BLACK)
            if open_ends == 2:
                threes += 1

    # 2. 4-4 금수
    if fours >= 2:
        return True, "4-4 금수"

    # 3. 3-3 금수
    if threes >= 2:
        return True, "3-3 금수"

    return False, ""
