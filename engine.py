# engine.py
import numpy as np

class STCheckerStrictEngine:
    def __init__(self):
        self.base_size = 10
        self.inventory = {"straight": 2, "diagonal": 2, "branch": float('inf')}
        self.hand_pieces = ["King"]
        self.game_clock = 300.0
        self.board_pieces = {}
        
        # [룰북 1] 플레이어 초기 기물
        for i in range(8):
            self.board_pieces[(8, i+1)] = {"type": "Checker", "stack": 0, "team": "player", "abyss": 0}
        self.board_pieces[(9, 3)] = {"type": "Knight", "stack": 0, "team": "player", "abyss": 0}
        self.board_pieces[(9, 6)] = {"type": "Knight", "stack": 0, "team": "player", "abyss": 0}
        
        # [룰북 1] AI 초기 기물
        for i in range(8):
            self.board_pieces[(1, i+1)] = {"type": "Checker", "stack": 0, "team": "enemy", "abyss": 0}
        self.board_pieces[(0, 3)] = {"type": "Knight", "stack": 0, "team": "enemy", "abyss": 0}
        self.board_pieces[(0, 6)] = {"type": "Knight", "stack": 0, "team": "enemy", "abyss": 0}

    def get_dynamic_board_size(self):
        """ [룰북 6] n = max(8, 10 + 손패 수 + 전장 최고 스택 수) 수축 공식 """
        max_stack = max([p["stack"] for p in self.board_pieces.values()]) if self.board_pieces else 0
        return max(8, 10 - max_stack)

    def generate_path(self, start, end):
        sx, sy = start
        ex, ey = end
        path = []
        steps = max(abs(ex - sx), abs(ey - sy))
        if steps == 0: return path
        dx = (ex - sx) // steps
        dy = (ey - sy) // steps
        for i in range(1, steps + 1):
            path.append((sx + dx * i, sy + dy * i))
        return path

    def execute_move(self, start, end):
        """ [룰북 3] 스택 제한 관통 사살 및 실시간 스택 누적 진화 """
        if start not in self.board_pieces: return False, "기물 없음"
        attacker = self.board_pieces[start]
        path = self.generate_path(start, end)
        
        current_stack = attacker["stack"]
        kill_count = 0
        final_pos = start

        for step in path:
            if step in self.board_pieces:
                target = self.board_pieces[step]
                if target["team"] != attacker["team"]:
                    if kill_count < current_stack:
                        kill_count += 1
                        current_stack += 1  # 실시간 스택 흡수 및 허용치 즉시 증가
                        del self.board_pieces[step]
                        final_pos = step
                    else:
                        break  # 한계 초과 시 저지됨
                else:
                    break
            else:
                final_pos = step

        if final_pos == start: return False, "이동 불가"

        attacker["stack"] = current_stack
        if current_stack >= 5: attacker["type"] = "🎴God"
        elif current_stack == 4: attacker["type"] = "👑Arch"
        elif current_stack == 3: attacker["type"] = "⚔️Adv"

        self.board_pieces[final_pos] = attacker
        del self.board_pieces[start]
        return True, f"성공 (스택 {current_stack})"
