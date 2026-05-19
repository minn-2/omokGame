import numpy as np

class Board:
    # [생성자] 게임 보드의 초기 설정 정의
    def __init__(self, board_size=15): 
        self.board_size = board_size    # 보드 크기 설정 (기본 15x15)
        self.reset()    # 게임 시작 상태로 초기화 호출

    # [초기화] 게임판을 비우고 상태를 리셋
    def reset(self):
        # 2차원 넘파이 배열 생성 | 0: 빈칸, 1: 흑돌, 2: 백돌
        self.board          = np.zeros(
            (self.board_size, self.board_size), dtype=int) 
        self.current_player = 1  # 렌주룰에 따라 흑돌(1)이 항상 선공
        self.is_over        = False # 게임 종료 플래그 (True 시 착수 중단)
        self.winner         = None # 승리자 정보 (None: 진행 중, 0: 무승부, 1: 흑, 2: 백)
        return self.get_state() # 현재 보드의 상태(데이터) 반환

    # [데이터 변환] 인공지능(AI) 모델의 입력값으로 쓰기 위한 변환
    def get_state(self):
        # 딥러닝 모델이 처리하기 쉬운 (C, H, W) 형태의 3차원 배열로 변환
        return (
            self.board.copy()
            .reshape(1, self.board_size, self.board_size)
            .astype(np.float32) # 연산 효율을 위해 실수형(float32)으로 캐스팅
        )

    # [착수 위치 계산] 현재 보드에서 돌을 놓을 수 있는 빈 공간 탐색
    def get_valid_moves(self):
        # board 배열에서 값이 0인 모든 좌표(Index)를 리스트 형태로 반환
        return np.argwhere(self.board == 0) 

    # [핵심 로직] 실제로 돌을 놓는 일련의 과정 처리
    def make_move(self, row, col):  
        # 1. 착수 예외 처리: 이미 돌이 있거나 게임이 이미 끝났다면 착수 거부
        if self.board[row, col] != 0 or self.is_over:
            return False

        # 2. 규칙 모듈 로드: 승패 및 금수 판정을 위해 외부 Rules 클래스 참조
        from core.Rules import Rules 
        
        # 3. 흑돌 전용 금수 검사: 흑돌(1)일 경우 착수 전 금수 자리인지 확인
        if self.current_player==1:
           if Rules.is_forbidden(
                self.board, row, col, self.current_player): 
            return False # 금수 자리라면 착수하지 않고 False 반환

        # 4. 돌 놓기: 검증이 끝난 위치에 현재 플레이어의 돌을 배치
        self.board[row, col] = self.current_player
        
        # 5. 승패 판정: Rules 클래스의 check_win을 통해 승리 조건 충족 확인
        if Rules.check_win(self.board, row, col):
            self.is_over = True # 승리 시 게임 종료 처리
            self.winner  = self.current_player # 현재 플레이어를 승자로 기록
            
        # 6. 무승부 판정: 보드에 더 이상 빈칸(0)이 없으면 무승부 처리
        elif not np.any(self.board == 0):
            self.is_over = True
            self.winner  = 0 # 승리자 없음(무승부)
            
        # 7. 턴 교체: 3에서 현재 번호를 빼서 다음 플레이어로 전환 (1->2, 2->1)
        self.current_player = 3 - self.current_player
        return True
