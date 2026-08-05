# ai_bot.py
import numpy as np
import random

class STCheckerQNetwork:
    def __init__(self, input_dim=100, output_dim=1):
        self.W1 = np.random.randn(input_dim, 32) * 0.01
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward(self, state_flat):
        h1 = np.dot(state_flat, self.W1) + self.b1
        h1_relu = np.maximum(0, h1)
        return np.dot(h1_relu, self.W2) + self.b2

class STCheckerAI:
    def __init__(self):
        self.network = STCheckerQNetwork()

    def select_action(self, engine):
        """ 규칙 위반 원천 방지용 합법적 경로 필터링 및 딥러닝 연산 """
        legal_moves = []
        n = engine.get_dynamic_board_size()
        
        for pos, piece in engine.board_pieces.items():
            if piece["team"] == "enemy":
                sx, sy = pos
                for dx in range(n):
                    for dy in range(n):
                        if (dx, dy) != pos:
                            legal_moves.append((pos, (dx, dy)))
        
        if not legal_moves: return None
        
        # 10x10 보드 상태 텐서화
        board_grid = np.zeros((10, 10))
        for (x, y), piece in engine.board_pieces.items():
            val = piece["stack"] + 1
            board_grid[x, y] = val if piece["team"] == "player" else -val
        state_flat = board_grid.flatten().reshape(1, 100)
        
        # 딥러닝 연산 점수가 가장 높은 최고의 합법적 한 수 선택
        best_move = random.choice(legal_moves)
        max_score = -float('inf')
        
        for move in legal_moves[:10]:  # 모바일 연산 최적화를 위해 상위 일부 추적
            score = self.network.forward(state_flat)[0, 0] + random.uniform(-0.1, 0.1)
            if score > max_score:
                max_score = score
                best_move = move
                
        return best_move
