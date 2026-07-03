import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List

# Импортируем готовые фичи и логику из твоего файла logic.py
from logic import _candidate_features, DIRECTIONS, HUNGRY_THRESHOLD, _in_bounds

class BattlesnakeRLEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 11
        self.height = 11
        
        # 4 действия: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        
        # 68 фичей: (13 базовых из logic.py + 4 кастомных пространственных) * 4 направления
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(68,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid_x, mid_y = 5, 5 # Центр поля 11х11
        self.snake_body = [(mid_x, mid_y), (mid_x, mid_y - 1), (mid_x, mid_y - 2)]
        self.health = 100
        
        # Спавним 3 случайные еды
        self.food = []
        for _ in range(3):
            self._spawn_food()
            
        self.turn = 0
        return self._get_obs(), {}

    def _spawn_food(self):
        while True:
            f = (random.randint(0, 10), random.randint(0, 10))
            if f not in self.snake_body and f not in self.food:
                self.food.append(f)
                break

    def _get_closest_item_vector(self, head, items):
        """Возвращает нормализованные dx, dy до ближайшего объекта"""
        if not items:
            return 0.0, 0.0
        closest = min(items, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
        dx = (closest[0] - head[0]) / 11.0
        dy = (closest[1] - head[1]) / 11.0
        return dx, dy

    def _get_game_state_dict(self) -> Dict:
        formatted_body = [{"x": p[0], "y": p[1]} for p in self.snake_body]
        formatted_food = [{"x": f[0], "y": f[1]} for f in self.food]
        you = {
            "id": "rl-snake", "name": "RL Snake", "health": self.health,
            "body": formatted_body, "head": formatted_body[0], "length": len(self.snake_body),
            "shout": "", "squad": ""
        }
        return {
            "game": {"id": "training", "ruleset": {"name": "standard"}, "timeout": 500},
            "turn": self.turn,
            "board": {"height": 11, "width": 11, "food": formatted_food, "hazards": [], "snakes": [you]},
            "you": you
        }

    def _get_obs(self) -> np.ndarray:
        state = self._get_game_state_dict()
        obs_vector = []
        
        # Порядок фичей из твоего файла logic.py
        feature_order = [
            "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
            "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
            "food_score", "food_delta", "is_food", "dist_to_center"
        ]
        
        for act_idx in range(4):
            move = self.action_map[act_idx]
            dx, dy = DIRECTIONS[move]
            nxt = (self.snake_body[0][0] + dx, self.snake_body[0][1] + dy)
            
            # Если ход суицидальный, забиваем блок фичей нулями
            if not _in_bounds(nxt, 11, 11) or nxt in self.snake_body[:-1]:
                obs_vector.extend([0.0] * 17)
                continue
                
            try:
                feats = _candidate_features(state, move)
                
                # Добавляем 13 базовых фичей (нормализуем их под 11х11)
                for name in feature_order:
                    val = float(feats.get(name, 0.0))
                    if name in ["space_capped", "open_space", "voronoi"]:
                        val /= 121.0 # Делим на площадь поля
                    elif name in ["wall_dist", "dist_to_center"]:
                        val /= 11.0  # Делим на линейный размер
                    obs_vector.append(val)
                
                # Добавляем 4 пространственные фичи (куда лететь?)
                food_dx, food_dy = self._get_closest_item_vector(nxt, self.food)
                obs_vector.extend([food_dx, food_dy, 0.0, 0.0]) # В симуляции врагов пока нет, пишем 0
                
            except Exception:
                obs_vector.extend([0.0] * 17)
                
        return np.array(obs_vector, dtype=np.float32)

    def step(self, action):
        self.turn += 1
        move = self.action_map[action]
        dx, dy = DIRECTIONS[move]
        nxt = (self.snake_body[0][0] + dx, self.snake_body[0][1] + dy)
        
        reward = 0.1 # Награда за выживание
        terminated, truncated = False, False
        
        if not _in_bounds(nxt, 11, 11) or nxt in self.snake_body[:-1]:
            return self._get_obs(), -100.0, True, False, {}
            
        self.snake_body.insert(0, nxt)
        self.health -= 1
        
        if nxt in self.food:
            reward = 15.0 # Скушал еду — молодец!
            self.health = 100
            self.food.remove(nxt)
            self._spawn_food()
        else:
            self.snake_body.pop()
            
        if self.health <= 0:
            return self._get_obs(), -50.0, True, False, {}
        if self.turn >= 400:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}