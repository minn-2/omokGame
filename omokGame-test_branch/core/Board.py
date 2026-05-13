import numpy as np

class Board:
    # 보드 생성
    def __init__(self, board_size=15):
        self.board_size = board_size # 보드 크기 저장
        self.reset() # 게임 초기화 호출
 
    # 게임 초기화
    def reset(self):
        self.board          = np.zeros( # 0 : 빈칸, 1 : 흑돌, 2 : 백돌
            (self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 흑돌 선공
        self.is_over        = False # 게임 종료 여부
        self.winner         = None # 승자 저장  none : 없음, 0 : 빈칸, 1 : 흑돌 승, 2 : 백돌 승
        return self.get_state() # 현재 상태 반환 호출

    # 현재 보드 상태 반환
    def get_state(self):
        return self.board.copy()\
            .reshape(1, self.board_size, self.board_size)\
            .astype(np.float32)

     # 가능한 수 반환
    def get_valid_moves(self):
        return np.argwhere(self.board == 0) # 값이 0인 위치 반환
        
    # 돌 놓기
    def make_move(self, row, col):
        # 이미 돌이 있거나 게임 종료 상태면 실패
        if self.board[row, col] != 0 or self.is_over:
            return False

        from core.Rules import Rules # Rules 클래스 가져오기
        
        # 금수 검사(흑돌만 적용)
        if Rules.is_forbidden(
                self.board, row, col, self.current_player): # 현재 플레이어 돌 놓기
            return False

        self.board[row, col] = self.current_player
        
        # 승리 판정
        if Rules.check_win(self.board, row, col):
            self.is_over = True # 게임 종료
            self.winner  = self.current_player # 승자 저장
            
        # 빈칸이 없으면 무승부
        elif not np.any(self.board == 0):
            self.is_over = True
            self.winner  = 0 # 무승부
            
        # 플레이어 교체
        self.current_player = 3 - self.current_player
        return True