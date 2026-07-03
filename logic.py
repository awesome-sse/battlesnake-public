"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#50C878",
        "head": "nr-rocket",
        "tail": "pixel",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Оркестратор (Идеальный симбиоз): 
    Глобально стратегию выбирает МЛ, но Лукахед проверяет её безопасность.
    """
    try:
        # 1. Запрашиваем ход у нашей линейной МЛ-модели
        ml_move = choose_move_model(game_state)
        
        if ml_move:
            # --- СИМБИОЗ: Валидация хода ручным алгоритмом ---
            # Проверяем ход модели через 2-шаговый просмотр вперед
            if _is_move_safe_lookahead(ml_move, game_state):
                return ml_move
            else:
                print(f">>> [СИМБИОЗ] МЛ выбрал ход '{ml_move}', но Лукахед обнаружил ловушку! Переключаем на ручной режим. <<<")
    except Exception:  
        pass

    try:
        # 2. Если МЛ-ход забракован или произошла ошибка — управление перехватывает ручной Лукахед
        lookahead_move = choose_move_lookahead(game_state)
        if lookahead_move:
            return lookahead_move
    except Exception:
        pass

    # 3. Финальный железный щит
    return choose_move_heuristic(game_state)


import json
import os

# Загружаем веса нейросети при старте сервера (работает мгновенно)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ppo_weights.json")
with open(WEIGHTS_PATH, "r") as f:
    _PPO_MODEL = json.load(f)

# КРИТИЧЕСКИ ВАЖНО: Узнайте, в каком порядке ваша среда возвращает действия!
# Обычно в Gym это: 0="up", 1="down", 2="left", 3="right" (проверьте в snake_env.py)
ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Вместо линейной модели запускает обученную MLP нейросеть [256, 128, 64]"""
    legal = _legal_moves(game_state)
    if not legal:
        return None

    # --- 1. Формируем вектор состояния (Observation Vector) ---
    # ВНИМАНИЕ: Сюда нужно передать точно такой же вектор/матрицу,
    # какую ваша среда BattlesnakeRLEnv генерирует внутри метода reset() или step().
    # Если ваша среда принимает на вход те самые 13 фич, код будет таким:
    # (Если среда принимает карту 11х11, вам нужно будет сформировать ее здесь)
    
    # Пример для 13 фич (возьмем дефолтный ход "up" просто для генерации базовых фич):
    feats_dict = _candidate_features(game_state, "up")
    features_names = [
        "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
        "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
        "food_score", "food_delta", "is_food", "dist_to_center"
    ]
    obs_vector = [feats_dict.get(name, 0.0) for name in features_names]

    # --- 2. Forward Pass (Нейросеть на чистом Python) ---
    def relu(vector):
        return [max(0.0, x) for x in vector]

    def layer_forward(x, W, b):
        output = []
        for row, bias in zip(W, b):
            output.append(sum(x_i * w_i for x_i, w_i in zip(x, row)) + bias)
        return output

    # Прогоняем через 3 скрытых слоя вашей архитектуры [256, 128, 64]
    h0 = relu(layer_forward(obs_vector, _PPO_MODEL["W_hidden_0"], _PPO_MODEL["b_hidden_0"]))
    h1 = relu(layer_forward(h0, _PPO_MODEL["W_hidden_1"], _PPO_MODEL["b_hidden_1"]))
    h2 = relu(layer_forward(h1, _PPO_MODEL["W_hidden_2"], _PPO_MODEL["b_hidden_2"]))
    
    # Получаем финальные 4 логита (оценки для каждого из 4 направлений)
    logits = layer_forward(h2, _PPO_MODEL["W_action"], _PPO_MODEL["b_action"])

    # --- 3. Выбираем лучшее разрешенное действие ---
    best_move = None
    best_score = float("-inf")
    
    for action_idx, move_name in ACTION_MAP.items():
        if move_name in legal: # Выбираем только среди безопасных ходов!
            score = logits[action_idx]
            if score > best_score:
                best_score = score
                best_move = move_name

    return best_move


def choose_move_lookahead(game_state: Dict) -> Optional[str]:
    """Безопасный ручной алгоритм с 2-шаговым просмотром пространства."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    
    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    legal_moves = _legal_moves(game_state)
    if not legal_moves:
        return None

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move in legal_moves:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)

        # ШАГ 1: Текущее пространство
        immediate_space = _flood_fill(nxt, occupied, width, height, limit=my_length + 2)
        move_score = immediate_space - 5000 if immediate_space <= my_length else float(immediate_space)

        if nxt in danger:
            move_score -= HEAD_TO_HEAD_PENALTY

        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            move_score += (width + height - nearest) * 5

        # ШАГ 2: Просмотр на шаг вперед (Lookahead)
        simulated_occupied = occupied.copy()
        if len(you["body"]) > 0:
            my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
            if my_tail in simulated_occupied:
                simulated_occupied.remove(my_tail)

        max_next_space = 0
        has_safe_escape = False

        for _, (ndx, ndy) in DIRECTIONS.items():
            future_cell = (nxt[0] + ndx, nxt[1] + ndy)
            if _in_bounds(future_cell, width, height) and future_cell not in simulated_occupied:
                future_space = _flood_fill(future_cell, simulated_occupied, width, height, limit=my_length + 2)
                if future_space > max_next_space:
                    max_next_space = future_space
                if future_space > my_length:
                    has_safe_escape = True

        if not has_safe_escape and max_next_space <= my_length:
            move_score -= 8000
        else:
            move_score += max_next_space

        if move_score > best_score:
            best_score = move_score
            best_move = move

    return best_move


def _is_move_safe_lookahead(move: str, game_state: Dict) -> bool:
    """Вспомогательная функция проверки: безопасен ли конкретный ход МЛ."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    
    occupied = _occupied_cells(board["snakes"])
    if nxt in occupied or not _in_bounds(nxt, width, height):
        return False
        
    # Проверка доступного места сразу
    immediate_space = _flood_fill(nxt, occupied, width, height, limit=my_length + 2)
    if immediate_space <= my_length:
        return False
        
    # Моделируем симуляцию освобождения хвоста
    simulated_occupied = occupied.copy()
    if len(you["body"]) > 0:
        my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
        if my_tail in simulated_occupied:
            simulated_occupied.remove(my_tail)
            
    # Проверяем, будет ли хоть один безопасный выход на следующем ходу
    has_safe_escape = False
    for _, (ndx, ndy) in DIRECTIONS.items():
        future_cell = (nxt[0] + ndx, nxt[1] + ndy)
        if _in_bounds(future_cell, width, height) and future_cell not in simulated_occupied:
            future_space = _flood_fill(future_cell, simulated_occupied, width, height, limit=my_length + 2)
            if future_space > my_length:
                has_safe_escape = True
                break
                
    return has_safe_escape


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or "up"


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


# --- Model -----------------------------------------------------

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}