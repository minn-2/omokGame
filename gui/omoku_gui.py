# ============================================================
# gomoku_gui.py  —  오목 GUI 모듈 (tkinter)
# ============================================================
# 실행 방법:
#   python gomoku_gui.py
#
# 필요 패키지: 파이썬 표준 라이브러리만 사용 (tkinter 내장)
# ============================================================

import tkinter as tk
from tkinter import messagebox, font as tkfont
from omoku_game import GomokuGame
from omoku_rules import BLACK, WHITE, EMPTY

# ── 상수 ────────────────────────────────────────────────────
BOARD_SIZE   = 15          # 15×15
CELL         = 40          # 격자 간격 (px)
MARGIN       = 40          # 보드 여백
STONE_R      = 16          # 돌 반지름
BOARD_PX     = MARGIN * 2 + CELL * (BOARD_SIZE - 1)   # 캔버스 크기

# 색상 팔레트
C_BG         = "#2B1D0E"   # 창 배경 (진한 갈색)
C_BOARD      = "#DEB887"   # 오목판 나무색
C_LINE       = "#8B6914"   # 격자선
C_BLACK      = "#111111"   # 흑돌
C_WHITE      = "#F5F5F0"   # 백돌
C_DOT        = "#5C3A1E"   # 화점(star point)
C_LAST_MARK  = "#E53935"   # 마지막 착수 표시
C_FORBIDDEN  = "#FF6B6B"   # 금수 표시 (잠깐 깜박)
C_STATUS_BG  = "#1A0F06"   # 상태바 배경
C_STATUS_FG  = "#FFD700"   # 상태바 텍스트


class GomokuGUI:
    """오목 GUI 메인 클래스."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("오목 (Gomoku)  ●  Human vs Human")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        self.game = GomokuGame(size=BOARD_SIZE, use_forbidden=True)
        self._build_ui()
        self._draw_board()
        self._update_status()

    # ── UI 빌드 ──────────────────────────────────────────────

    def _build_ui(self):
        # 상단 타이틀
        title_font = tkfont.Font(family="Georgia", size=18, weight="bold")
        tk.Label(
            self.root, text="오  목", font=title_font,
            bg=C_BG, fg=C_STATUS_FG
        ).pack(pady=(16, 4))

        # 캔버스
        self.canvas = tk.Canvas(
            self.root,
            width=BOARD_PX, height=BOARD_PX,
            bg=C_BOARD, highlightthickness=0
        )
        self.canvas.pack(padx=20)
        self.canvas.bind("<Button-1>", self._on_click)

        # 상태 레이블
        status_font = tkfont.Font(family="Malgun Gothic", size=13, weight="bold")
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=status_font, bg=C_STATUS_BG, fg=C_STATUS_FG,
            relief="flat", padx=12, pady=6, anchor="center"
        )
        self.status_label.pack(fill="x", padx=20, pady=(8, 4))

        # 버튼 영역
        btn_frame = tk.Frame(self.root, bg=C_BG)
        btn_frame.pack(pady=(4, 16))

        btn_style = dict(
            font=tkfont.Font(family="Malgun Gothic", size=11),
            relief="flat", cursor="hand2",
            padx=16, pady=6
        )
        tk.Button(
            btn_frame, text="⟳  새 게임", bg="#4CAF50", fg="white",
            command=self._new_game, **btn_style
        ).pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="↩  무르기", bg="#2196F3", fg="white",
            command=self._undo, **btn_style
        ).pack(side="left", padx=6)

        tk.Button(
            btn_frame, text="✕  종료", bg="#F44336", fg="white",
            command=self.root.quit, **btn_style
        ).pack(side="left", padx=6)

    # ── 보드 그리기 ──────────────────────────────────────────

    def _draw_board(self):
        """격자, 화점, 돌을 모두 다시 그린다."""
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_star_points()
        self._draw_stones()

    def _draw_grid(self):
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL
            y = MARGIN + i * CELL
            # 세로선
            self.canvas.create_line(
                x, MARGIN, x, MARGIN + (BOARD_SIZE - 1) * CELL,
                fill=C_LINE, width=1
            )
            # 가로선
            self.canvas.create_line(
                MARGIN, y, MARGIN + (BOARD_SIZE - 1) * CELL, y,
                fill=C_LINE, width=1
            )

    def _draw_star_points(self):
        """화점 위치: 표준 15×15 기준"""
        star_points = [3, 7, 11]   # 0-based 인덱스
        for r in star_points:
            for c in star_points:
                cx = MARGIN + c * CELL
                cy = MARGIN + r * CELL
                self.canvas.create_oval(
                    cx - 4, cy - 4, cx + 4, cy + 4,
                    fill=C_DOT, outline=""
                )

    def _draw_stones(self):
        """현재 보드 상태의 모든 돌을 그린다."""
        history = self.game.move_history
        last = (history[-1][0], history[-1][1]) if history else None

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = self.game.board[r][c]
                if cell == EMPTY:
                    continue
                cx = MARGIN + c * CELL
                cy = MARGIN + r * CELL
                color  = C_BLACK if cell == BLACK else C_WHITE
                border = "#444" if cell == BLACK else "#AAA"
                self.canvas.create_oval(
                    cx - STONE_R, cy - STONE_R,
                    cx + STONE_R, cy + STONE_R,
                    fill=color, outline=border, width=1
                )
                # 마지막 착수 빨간 점
                if last and (r, c) == last:
                    dot_color = C_WHITE if cell == BLACK else C_BLACK
                    self.canvas.create_oval(
                        cx - 4, cy - 4, cx + 4, cy + 4,
                        fill=C_LAST_MARK, outline=""
                    )

    # ── 이벤트 핸들러 ────────────────────────────────────────

    def _on_click(self, event):
        if self.game.game_over:
            return

        # 클릭 좌표 → 가장 가까운 교차점
        col = round((event.x - MARGIN) / CELL)
        row = round((event.y - MARGIN) / CELL)

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        result = self.game.place_stone(row, col)

        if not result["success"]:
            if result["forbidden"]:
                self._flash_forbidden(row, col, result["forbidden_reason"])
            else:
                self._update_status(result["message"], color="#FF8A65")
            return

        self._draw_board()
        self._update_status(result["message"])

        if self.game.game_over:
            self._on_game_over(result["message"])

    def _on_game_over(self, message: str):
        """게임 종료 처리."""
        self._update_status(message, color="#FFD700")
        answer = messagebox.askyesno(
            "게임 종료",
            f"{message}\n\n새 게임을 시작하시겠습니까?",
            parent=self.root
        )
        if answer:
            self._new_game()

    def _new_game(self):
        self.game.reset()
        self._draw_board()
        self._update_status()

    def _undo(self):
        if self.game.undo():
            self._draw_board()
            self._update_status("무르기 완료.", color="#80DEEA")
        else:
            self._update_status("무를 수 있는 수가 없습니다.", color="#FF8A65")

    # ── 금수 깜박 효과 ───────────────────────────────────────

    def _flash_forbidden(self, row, col, reason, count=3):
        """금수 위치에 빨간 원을 잠깐 표시한다."""
        cx = MARGIN + col * CELL
        cy = MARGIN + row * CELL

        def blink(n):
            if n <= 0:
                self.canvas.delete("forbidden_mark")
                self._update_status(
                    f"금수! ({reason}) — 다른 위치에 놓으세요.",
                    color="#FF6B6B"
                )
                return
            self.canvas.delete("forbidden_mark")
            if n % 2 == 1:
                self.canvas.create_oval(
                    cx - STONE_R, cy - STONE_R,
                    cx + STONE_R, cy + STONE_R,
                    fill=C_FORBIDDEN, outline="red", width=2,
                    tags="forbidden_mark"
                )
            self.root.after(180, lambda: blink(n - 1))

        blink(count * 2)

    # ── 상태 업데이트 ────────────────────────────────────────

    def _update_status(self, extra: str = "", color: str = C_STATUS_FG):
        player = self.game.current_player_name()
        if self.game.game_over:
            text = extra or "게임 종료"
        else:
            text = f"차례: {player}  {('— ' + extra) if extra else ''}"
        self.status_var.set(text)
        self.status_label.config(fg=color)


# ── 진입점 ───────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
