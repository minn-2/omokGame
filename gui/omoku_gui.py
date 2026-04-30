# ============================================================
# omoku_gui.py  —  오목 GUI 모듈 (tkinter)
# ============================================================
# 실행 방법:
#   python omoku_gui.py
#
# 기능:
#   - 시작 화면에서 모드 선택 (Human vs Human / Human vs AI)
#   - Human vs Human: 두 사람이 번갈아 착수
#   - Human vs AI: 플레이어(흑) vs AI(백)
#   - 금수 깜박 효과, 무르기, 새 게임, 메뉴로 돌아가기
# ============================================================

import tkinter as tk
from tkinter import messagebox, font as tkfont
from core.omoku_game import GomokuGame
from core.omoku_rules import BLACK, WHITE, EMPTY
from ai.omoku_ai import OmokuAI

# ── 상수 ────────────────────────────────────────────────────
BOARD_SIZE  = 15
CELL        = 40
MARGIN      = 40
STONE_R     = 16
BOARD_PX    = MARGIN * 2 + CELL * (BOARD_SIZE - 1)

C_BG        = "#1C1207"
C_BOARD     = "#D4A96A"
C_LINE      = "#7A5C1E"
C_BLACK     = "#0D0D0D"
C_WHITE     = "#F2EFE4"
C_DOT       = "#4A2E0A"
C_LAST_MARK = "#E53935"
C_FORBIDDEN = "#FF6B6B"
C_STATUS_BG = "#120C04"
C_STATUS_FG = "#FFD700"
C_AI_BADGE  = "#00BCD4"

MODE_HVH = "human_vs_human"
MODE_HVA = "human_vs_ai"


# ════════════════════════════════════════════════════════════
# 시작 화면
# ════════════════════════════════════════════════════════════

class StartScreen:
    def __init__(self, root, on_start):
        self.root = root
        self.on_start = on_start
        self.root.title("오목 — 모드 선택")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)
        self._build()

    def _build(self):
        wrapper = tk.Frame(self.root, bg=C_BG)
        wrapper.pack(expand=True, padx=60, pady=50)

        title_font = tkfont.Font(family="Georgia", size=32, weight="bold")
        tk.Label(wrapper, text="오  목", font=title_font,
                 bg=C_BG, fg=C_STATUS_FG).pack(pady=(0, 6))

        sub_font = tkfont.Font(family="Malgun Gothic", size=11)
        tk.Label(wrapper, text="Gomoku — 모드를 선택하세요",
                 font=sub_font, bg=C_BG, fg="#A0896A").pack(pady=(0, 36))

        btn_font     = tkfont.Font(family="Malgun Gothic", size=13, weight="bold")
        sub_btn_font = tkfont.Font(family="Malgun Gothic", size=9)

        # Human vs Human
        hvh_frame = tk.Frame(wrapper, bg="#2A1A08")
        hvh_frame.pack(fill="x", pady=8, ipady=4)
        tk.Button(hvh_frame, text="👥  Human  vs  Human",
                  font=btn_font, bg="#4E3416", fg="#FFD700",
                  activebackground="#6B4A22", activeforeground="#FFE57F",
                  relief="flat", cursor="hand2", padx=20, pady=14,
                  command=lambda: self._select(MODE_HVH)).pack(fill="x")
        tk.Label(hvh_frame,
                 text="두 사람이 번갈아 대국합니다  (흑 선공, 금수 적용)",
                 font=sub_btn_font, bg="#2A1A08", fg="#8C7050").pack(pady=(0, 6))

        # Human vs AI
        hva_frame = tk.Frame(wrapper, bg="#0A1E2A")
        hva_frame.pack(fill="x", pady=8, ipady=4)
        tk.Button(hva_frame, text="🤖  Human  vs  AI",
                  font=btn_font, bg="#0D2D42", fg=C_AI_BADGE,
                  activebackground="#1A4060", activeforeground="#80DEEA",
                  relief="flat", cursor="hand2", padx=20, pady=14,
                  command=lambda: self._select(MODE_HVA)).pack(fill="x")
        tk.Label(hva_frame,
                 text="플레이어(흑) vs AI(백)  —  omoku_ai.py에서 AI 강화 가능",
                 font=sub_btn_font, bg="#0A1E2A", fg="#3A7A9A").pack(pady=(0, 6))

        tk.Button(wrapper, text="종료", font=sub_btn_font,
                  bg=C_BG, fg="#554433",
                  activebackground=C_BG, activeforeground="#AA6633",
                  relief="flat", cursor="hand2",
                  command=self.root.quit).pack(pady=(20, 0))

    def _select(self, mode):
        for w in self.root.winfo_children():
            w.destroy()
        self.on_start(mode)


# ════════════════════════════════════════════════════════════
# 게임 화면
# ════════════════════════════════════════════════════════════

class GomokuGUI:
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.ai   = OmokuAI(player=WHITE) if mode == MODE_HVA else None

        title = "오목  ●  Human vs Human" if mode == MODE_HVH else "오목  🤖  Human vs AI"
        self.root.title(title)
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        self.game = GomokuGame(size=BOARD_SIZE, use_forbidden=True)
        self._build_ui()
        self._draw_board()
        self._update_status()

    # ── UI 빌드 ──────────────────────────────────────────────

    def _build_ui(self):
        header = tk.Frame(self.root, bg=C_BG)
        header.pack(fill="x", padx=20, pady=(14, 2))

        title_font = tkfont.Font(family="Georgia", size=18, weight="bold")
        tk.Label(header, text="오  목", font=title_font,
                 bg=C_BG, fg=C_STATUS_FG).pack(side="left")

        badge_text  = "👥 Human vs Human" if self.mode == MODE_HVH else "🤖 Human vs AI"
        badge_color = "#4E3416"            if self.mode == MODE_HVH else "#0D2D42"
        badge_fg    = "#FFD700"            if self.mode == MODE_HVH else C_AI_BADGE
        tk.Label(header, text=badge_text,
                 font=tkfont.Font(family="Malgun Gothic", size=10),
                 bg=badge_color, fg=badge_fg, padx=10, pady=3).pack(side="right")

        self.canvas = tk.Canvas(self.root, width=BOARD_PX, height=BOARD_PX,
                                bg=C_BOARD, highlightthickness=2,
                                highlightbackground="#5C3A1E")
        self.canvas.pack(padx=20)
        self.canvas.bind("<Button-1>", self._on_click)

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=tkfont.Font(family="Malgun Gothic", size=12, weight="bold"),
            bg=C_STATUS_BG, fg=C_STATUS_FG,
            relief="flat", padx=12, pady=7, anchor="center")
        self.status_label.pack(fill="x", padx=20, pady=(8, 4))

        btn_frame = tk.Frame(self.root, bg=C_BG)
        btn_frame.pack(pady=(4, 16))
        bs = dict(font=tkfont.Font(family="Malgun Gothic", size=11),
                  relief="flat", cursor="hand2", padx=14, pady=6)

        tk.Button(btn_frame, text="⟳  새 게임",   bg="#4CAF50", fg="white",
                  command=self._new_game,       **bs).pack(side="left", padx=5)
        tk.Button(btn_frame, text="↩  무르기",    bg="#2196F3", fg="white",
                  command=self._undo,           **bs).pack(side="left", padx=5)
        tk.Button(btn_frame, text="🏠  모드 선택", bg="#FF9800", fg="white",
                  command=self._back_to_menu,   **bs).pack(side="left", padx=5)
        tk.Button(btn_frame, text="✕  종료",      bg="#F44336", fg="white",
                  command=self.root.quit,       **bs).pack(side="left", padx=5)

    # ── 보드 그리기 ──────────────────────────────────────────

    def _draw_board(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_star_points()
        self._draw_stones()

    def _draw_grid(self):
        end = MARGIN + (BOARD_SIZE - 1) * CELL
        for i in range(BOARD_SIZE):
            x = MARGIN + i * CELL
            y = MARGIN + i * CELL
            self.canvas.create_line(x, MARGIN, x, end, fill=C_LINE, width=1)
            self.canvas.create_line(MARGIN, y, end, y, fill=C_LINE, width=1)

    def _draw_star_points(self):
        for r in [3, 7, 11]:
            for c in [3, 7, 11]:
                cx = MARGIN + c * CELL
                cy = MARGIN + r * CELL
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4,
                                        fill=C_DOT, outline="")

    def _draw_stones(self):
        history = self.game.move_history
        last = (history[-1][0], history[-1][1]) if history else None

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = self.game.board[r][c]
                if val == EMPTY:
                    continue
                cx = MARGIN + c * CELL
                cy = MARGIN + r * CELL
                color  = C_BLACK if val == BLACK else C_WHITE
                border = "#333"  if val == BLACK else "#BBB"
                self.canvas.create_oval(cx-STONE_R, cy-STONE_R,
                                        cx+STONE_R, cy+STONE_R,
                                        fill=color, outline=border, width=1)
                if last and (r, c) == last:
                    if self.mode == MODE_HVA and val == WHITE:
                        self.canvas.create_text(cx, cy, text="🤖",
                                                font=("", 10))
                    else:
                        self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4,
                                                fill=C_LAST_MARK, outline="")

    # ── 이벤트 ───────────────────────────────────────────────

    def _on_click(self, event):
        if self.game.game_over:
            return
        if self.mode == MODE_HVA and self.game.current_player == WHITE:
            return

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
            return

        if self.mode == MODE_HVA and self.game.current_player == WHITE:
            self.root.after(400, self._ai_move)

    def _ai_move(self):
        if self.game.game_over:
            return
        self._update_status("AI 생각 중...", color=C_AI_BADGE)
        self.root.update()

        row, col = self.ai.get_move(self.game.get_board_copy())
        result   = self.game.place_stone(row, col)
        self._draw_board()
        self._update_status(result["message"])

        if self.game.game_over:
            self._on_game_over(result["message"])

    def _on_game_over(self, message):
        self._update_status(message, color="#FFD700")
        if messagebox.askyesno("게임 종료",
                               f"{message}\n\n새 게임을 시작하시겠습니까?",
                               parent=self.root):
            self._new_game()

    def _new_game(self):
        self.game.reset()
        self._draw_board()
        self._update_status()

    def _undo(self):
        if self.mode == MODE_HVA:
            self.game.undo()   # AI 수 취소
        self.game.undo()       # 내 수 취소
        self._draw_board()
        self._update_status("무르기 완료.", color="#80DEEA")

    def _back_to_menu(self):
        for w in self.root.winfo_children():
            w.destroy()
        StartScreen(self.root, lambda mode: GomokuGUI(self.root, mode))

    def _flash_forbidden(self, row, col, reason, count=3):
        cx = MARGIN + col * CELL
        cy = MARGIN + row * CELL
        def blink(n):
            if n <= 0:
                self.canvas.delete("forbidden_mark")
                self._update_status(f"금수! ({reason}) — 다른 위치에 놓으세요.",
                                    color="#FF6B6B")
                return
            self.canvas.delete("forbidden_mark")
            if n % 2 == 1:
                self.canvas.create_oval(cx-STONE_R, cy-STONE_R,
                                        cx+STONE_R, cy+STONE_R,
                                        fill=C_FORBIDDEN, outline="red",
                                        width=2, tags="forbidden_mark")
            self.root.after(180, lambda: blink(n - 1))
        blink(count * 2)

    def _update_status(self, extra="", color=C_STATUS_FG):
        if self.game.game_over:
            text = extra or "게임 종료"
        elif self.mode == MODE_HVA and self.game.current_player == WHITE:
            text  = "🤖 AI 차례 (백○)"
            color = C_AI_BADGE
        else:
            player = self.game.current_player_name()
            text   = f"차례: {player}  {('— ' + extra) if extra else ''}"
        self.status_var.set(text)
        self.status_label.config(fg=color)


# ── 진입점 ───────────────────────────────────────────────────

def main():
    root = tk.Tk()
    StartScreen(root, lambda mode: GomokuGUI(root, mode))
    root.mainloop()

if __name__ == "__main__":
    main()
