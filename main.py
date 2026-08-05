# main.py
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock

from engine import STCheckerStrictEngine
from ai_bot import STCheckerAI

class STCheckerPlayUI(BoxLayout):
    def __init__(self, **kwargs):
        super(STCheckerPlayUI, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.engine = STCheckerStrictEngine()
        self.ai = STCheckerAI()
        self.selected_tile = None
        self.turn = "player"

        self.status_bar = Label(text="[My Turn] Rules Active", size_hint_y=0.1, font_size='18sp')
        self.add_widget(self.status_bar)

        self.grid = GridLayout(cols=10, rows=10)
        self.buttons = {}
        self.build_board()
        self.add_widget(self.grid)
        Clock.schedule_interval(self.update_clock, 1.0)

    def build_board(self):
        self.grid.clear_widgets()
        n = self.engine.get_dynamic_board_size()

        for r in range(10):
            for c in range(10):
                piece = self.engine.board_pieces.get((r, c))
                if r >= n or c >= n:
                    btn = Button(text="🌌", background_color=(0.1, 0, 0.2, 1))
                else:
                    if piece:
                        color = "🔵" if piece["team"] == "player" else "🔴"
                        btn_text = f"{color}\n{piece['type'][:4]}({piece['stack']})"
                    else:
                        btn_text = ""
                    btn = Button(text=btn_text)
                    btn.bind(on_press=self.on_tile_press)
                btn.matrix_pos = (r, c)
                self.grid.add_widget(btn)
                self.buttons[(r, c)] = btn

    def on_tile_press(self, instance):
        if self.turn != "player": return
        pos = instance.matrix_pos
        
        if self.selected_tile is None:
            if pos in self.engine.board_pieces and self.engine.board_pieces[pos]["team"] == "player":
                self.selected_tile = pos
                instance.background_color = (0, 1, 0, 1)
        else:
            success, msg = self.engine.execute_move(self.selected_tile, pos)
            self.selected_tile = None
            self.build_board()
            
            if success:
                self.turn = "enemy"
                self.status_bar.text = f"[AI Turn] Calculating..."
                Clock.schedule_once(self.trigger_ai_turn, 1.0)
            else:
                self.status_bar.text = f"[Denied] {msg}"

    def trigger_ai_turn(self, dt):
        ai_move = self.ai.select_action(self.engine)
        if ai_move:
            start, end = ai_move
            self.engine.execute_move(start, end)
        self.build_board()
        self.turn = "player"
        self.status_bar.text = f"[My Turn] Size: {self.engine.get_dynamic_board_size()}x{self.engine.get_dynamic_board_size()}"

    def update_clock(self, dt):
        self.engine.game_clock -= 1.0
        if self.engine.game_clock <= 0:
            self.status_bar.text = "GAME OVER"
            self.turn = "gameover"

class STCheckerApp(App):
    def build(self):
        return STCheckerPlayUI()

if __name__ == '__main__':
    STCheckerApp().run()
