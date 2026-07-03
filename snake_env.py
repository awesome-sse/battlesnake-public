import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Set, Tuple

# Импортируем базовые функции
from logic import _candidate_features, DIRECTIONS, HUNGRY_THRESHOLD, _in_bounds, _flood_fill

Point = Tuple[int, int]

class BattlesnakeRLEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 11
        self.height = 11
        
        # ТЕПЕРЬ 5 ДЕЙСТВИЙ: 0-3 — направления, 4 — передача управления Лукахеду
        self.action_space = spaces.Discrete(5)
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "manual"}
        
        # 68 фичей (описание поля по 4 направлениям) остаются прежними
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(68,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid_x, mid_y = 5, 5
        self.snake_body = [(mid_x, mid_y), (mid_x, mid_y - 1), (mid_x, mid_y - 2)]
        self.health = 100
        
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
        if not items:
            return 0.0, 0.0
        closest = min(items, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
        return (closest[0] - head[0]) / 11.0, (closest[1] - head[1]) / 11.0

    def _get_game_state_dict(self) -> Dict:
        formatted_body = [{"x": p[0], "y": p[1]} for p in self.snake_body]
        formatted_food = [{"x": f[0], "y": f[1]} for f in self.food]
        you = {
            "id": "rl-snake", "name": "RL Snake", "health": self.health,
            "body": formatted_body, "head": formatted_body[0], "length": len(self.snake_body)
        }
        return {
            "game": {"id": "training"}, "turn": self.turn,
            "board": {"height": 11, "width": 11, "food": formatted_food, "hazards": [], "snakes": [you]},
            "you": you
        }

    def _get_obs(self) -> np.ndarray:
        state = self._get_game_state_dict()
        obs_vector = []
        feature_order = [
            "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
            "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
            "food_score", "food_delta", "is_food", "dist_to_center"
        ]
        for act_idx in range(4): # Фичи всегда описывают 4 физических направления
            move = self.action_map[act_idx]
            dx, dy = DIRECTIONS[move]
            nxt = (self.snake_body[0][0] + dx, self.snake_body[0][1] + dy)
            if not _in_bounds(nxt, 11, 11) or nxt in self.snake_body[:-1]:
                obs_vector.extend([0.0] * 17)
                continue
            try:
                feats = _candidate_features(state, move)
                for name in feature_order:
                    val = float(feats.get(name, 0.0))
                    if name in ["space_capped", "open_space", "voronoi"]: val /= 121.0
                    elif name in ["wall_dist", "dist_to_center"]: val /= 11.0
                    obs_vector.append(val)
                food_dx, food_dy = self._get_closest_item_vector(nxt, self.food)
                obs_vector.extend([food_dx, food_dy, 0.0, 0.0])
            except Exception:
                obs_vector.extend([0.0] * 17)
        return np.array(obs_vector, dtype=np.float32)

    def _env_lookahead(self) -> str:
        """Встроенный в среду Лукахед для симуляции 5-го действия."""
        head = self.snake_body[0]
        occupied = set(self.snake_body)
        best_move = "up"
        best_score = float("-inf")
        
        legal = [m for m, (dx, dy) in DIRECTIONS.items() if _in_bounds((head[0]+dx, head[1]+dy), 11, 11) and (head[0]+dx, head[1]+dy) not in occupied]
        if not legal: return "up"
        
        for move in legal:
            dx, dy = DIRECTIONS[move]
            nxt = (head[0] + dx, head[1] + dy)
            immediate_space = _flood_fill(nxt, occupied, 11, 11, limit=len(self.snake_body) + 2)
            score = immediate_space - 5000 if immediate_space <= len(self.snake_body) else float(immediate_space)
            
            if self.food and self.health < HUNGRY_THRESHOLD:
                nearest = min(abs(nxt[0]-f[0])+abs(nxt[1]-f[1]) for f in self.food)
                score += (22 - nearest) * 5
                
            sim_occupied = occupied.copy()
            if nxt not in self.food and len(self.snake_body) > 0:
                if self.snake_body[-1] in sim_occupied: sim_occupied.remove(self.snake_body[-1])
                
            max_next = 0
            for _, (ndx, ndy) in DIRECTIONS.items():
                f_cell = (nxt[0]+ndx, nxt[1]+ndy)
                if _in_bounds(f_cell, 11, 11) and f_cell not in sim_occupied:
                    f_space = _flood_fill(f_cell, sim_occupied, 11, 11, limit=len(self.snake_body) + 2)
                    if f_space > max_next: max_next = f_space
            score += max_next
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def step(self, action):
        self.turn += 1
        
        # СТРАТЕГИЧЕСКИЙ ВЫБОР: Если модель выбрала 4, отдаем шаг Лукахеду
        if action == 4:
            move = self._env_lookahead()
            reward_penalty = -0.05 # Небольшой штраф, чтобы модель не ленилась ходить сама без повода
        else:
            move = self.action_map[action]
            reward_penalty = 0.0
            
        dx, dy = DIRECTIONS[move]
        nxt = (self.snake_body[0][0] + dx, self.snake_body[0][1] + dy)
        
        reward = 0.1 + reward_penalty
        terminated, truncated = False, False
        
        if not _in_bounds(nxt, 11, 11) or nxt in self.snake_body[:-1]:
            return self._get_obs(), -100.0, True, False, {}
            
        self.snake_body.insert(0, nxt)
        self.health -= 1
        
        if nxt in self.food:
            reward = 15.0
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