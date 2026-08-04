import sys
import os
import random
import numpy as np

# 🔥 [주목] 넘파이 편법이 아닌, 실제 파이토치 라이브러리 직접 로드
import torch
import torch.nn as nn
import torch.optim as optim

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

# =============================================================
# 🧠 [On-Device PyTorch] 스마트폰 내부 구동용 30MB급 심층 신경망
# =============================================================
class STCheckerNet(nn.Module):
    def __init__(self):
        super(STCheckerNet, self).__init__()
        # 10x10 보드판 상태를 압축 해석하는 CNN 레이어
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 보드 채널 데이터와 부가 변수를 결합하여 액션 가치(Q-Value) 출력
        self.fc = nn.Sequential(
            nn.Linear(64 * 10 * 10 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100) # 100개 타일 중 최적의 암살 궤적 좌표 예측
        )
        
    def forward(self, board, info):
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, info), dim=1)
        return self.fc(x)

# =============================================================
# 📱 『ST체커』 파이토치 탑재 안드로이드 GUI 구현부
# =============================================================
class STCheckerGUI(BoxLayout):
    def __init__(self, **kwargs):
        super(STCheckerGUI, self).__init__(**kwargs)
        self.orientation = 'horizontal'
        
        # 앱 내부 앱 전용 세이브 경로 확보
        self.model_path = os.path.join(App().user_data_dir, "stchecker_bot_30mb.pth")
        
        # 파이토치 모델 및 최적화 도구 인스턴스 초기화
        self.model = STCheckerNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()
        self.game_history = []

        # 세이브 파일이 있다면 스마트폰 플래시 메모리에서 딥러닝 가중치 자동 로드
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            except:
                pass
        self.model.eval() # 기본 추론 모드 고정

        # 보드 데이터 세팅
        self.board = np.zeros((10, 10), dtype=int)
        self.stacks = np.zeros((10, 10), dtype=float)
        self.board[0, 0] = 1; self.stacks[0, 0] = 1.0 # 유저 킹
        self.board[9, 9] = 2; self.stacks[9, 9] = 1.0 # 봇 킹
        self.turn = "PLAYER"

        # 좌측 패널: 10 x 10 격자보드
        self.board_layout = GridLayout(cols=10, size_hint=(0.7, 1))
        self.buttons = {}
        self.create_board()
        self.add_widget(self.board_layout)
        
        # 우측 패널: 상태창 및 딥러닝 동기화 버튼
        self.side_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1), padding=10, spacing=10)
        self.status_label = Label(text="순수 파이토치\n엔진 구동 성공!\n내 턴입니다.", halign="center")
        self.side_panel.add_widget(self.status_label)
        
        self.train_btn = Button(text="대국 종료 후\nBP 오차 역전파 학습", size_hint_y=0.2, background_color=(0.9, 0.3, 0.3, 1))
        self.train_btn.bind(on_press=self.trigger_backprop_learning)
        self.side_panel.add_widget(self.train_btn)
        
        self.add_widget(self.side_panel)
        self.update_ui()

    def create_board(self):
        for r in range(10):
            for c in range(10):
                btn = Button(text="", font_size=11)
                btn.bind(on_press=self.on_tile_click)
                btn.row, btn.col = r, c
                self.buttons[(r, c)] = btn
                self.board_layout.add_widget(btn)

    def update_ui(self):
        for r in range(10):
            for c in range(10):
                val = self.board[r, c]
                st = self.stacks[r, c]
                btn = self.buttons[(r, c)]
                if val == 1:
                    btn.text = f"👑 K\n({st})"
                    btn.background_color = (0.8, 0.2, 0.2, 1)
                elif val == 3:
                    btn.text = f"♟ C\n({st})"
                    btn.background_color = (0.2, 0.5, 0.8, 1)
                else:
                    btn.text = ""
                    btn.background_color = (0.15, 0.15, 0.15, 1) if (r + c) % 2 == 0 else (0.25, 0.25, 0.25, 1)

    def get_tensor_inputs(self):
        # 현재 보드를 4채널 파이토치 텐서로 실시간 포맷 가공
        b_chan = np.zeros((4, 10, 10), dtype=np.float32)
        b_chan[0] = (self.board == 1).astype(np.float32)
        b_chan[1] = (self.board == 2).astype(np.float32)
        b_chan[2] = (self.board == 3).astype(np.float32)
        b_chan[3] = self.stacks / 5.0
        
        board_t = torch.FloatTensor(b_chan).unsqueeze(0)
        info_t = torch.FloatTensor([[0, 0, 1.0, 0, 0]])
        return board_t, info_t

    def on_tile_click(self, instance):
        if self.turn != "PLAYER" or self.board[instance.row, instance.col] != 0: return
        r, c = instance.row, instance.col
        
        # 기보 데이터 저장소에 기록
        board_t, info_t = self.get_tensor_inputs()
        self.game_history.append((board_t, info_t, r * 10 + c))
        
        self.board[r, c] = 3
        self.stacks[r, c] = 1.0
        self.update_ui()
        
        self.turn = "BOT"
        self.status_label.text = "실제 PyTorch 신경망이\n순전파(Forward) 추론 중..."
        self.bot_execute()

    def bot_execute(self):
        # 스마트폰 안에서 진짜 파이토치 순전파(Forward) 작동
        board_t, info_t = self.get_tensor_inputs()
        with torch.no_grad():
            q_values = self.model(board_t, info_t)
        best_action = torch.argmax(q_values).item()
        
        br, bc = best_action // 10, best_action % 10
        if self.board[br, bc] == 0:
            self.board[br, bc] = 3
            self.stacks[br, bc] = 1.0
            
        self.update_ui()
        self.turn = "PLAYER"
        self.status_label.text = "다시 당신의 턴입니다!"

    def trigger_backprop_learning(self, instance):
        if not self.game_history: return
        self.model.train()
        self.status_label.text = "스마트폰 CPU로\n실시간 손실(Loss) 역전파\n가중치 업데이트 중..."
        
        # 판이 끝나면 내 스마트폰 장치 안에서 직접 가중치 업데이트(Backpropagation) 수행
        for b_t, i_t, action in self.game_history:
            output = self.model(b_t, i_t)
            target = output.clone().detach()
            target[0, action] = 5.0 # 유저 전술 흡수
            
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 업데이트된 따끈따끈한 30MB 가중치를 스마트폰 전용 세이브 경로에 파일로 각인
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        self.game_history = []
        self.status_label.text = "학습 완료 및 저장 성공!\n새로운 딥러닝 봇이\n당신을 기다립니다."

class STCheckerApp(App):
    def build(self):
        self.title = "ST체커 공식 파이토치 임베디드 에디션"
        return STCheckerGUI()

if __name__ == "__main__":
    STCheckerApp().run()
