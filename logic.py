"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import os
from stable_baselines3 import PPO
import numpy as np

# Автозагрузка весов при запуске сервера Flask
MODEL_PATH = "ppo_battlesnake_11x11.zip"
if os.path.exists(MODEL_PATH):
    RL_MODEL = PPO.load(MODEL_PATH)
    print(">>> ИИ-модель PPO успешно подключена и готова к бою! <<<")
else:
    RL_MODEL = None
    print(">>> ВНИМАНИЕ: Файл модели не найден. Работает хвристический фолбек! <<<")

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
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the model, with a heuristic fallback."""
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


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

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body.

    We keep tails occupied too; they only free up *next* turn and treating them
    as solid is the conservative, safe choice for a base bot.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
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
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
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
    """Shortest free-cell distances from seed cells."""
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
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
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

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
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
# Embedded standardized linear model.

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


def choose_move_model(game_state: Dict) -> Optional[str]:
    if RL_MODEL is None:
        return None

    legal = _legal_moves(game_state)
    if not legal:
        return None

    # Извлекаем инфо о текущем состоянии для пространственных фичей
    head = (game_state["you"]["body"][0]["x"], game_state["you"]["body"][0]["y"])
    
    # Список координат всей еды на поле
    foods = [(f["x"], f["y"]) for f in game_state["board"]["food"]]
    
    # Список координат голов ВСЕХ врагов
    enemies = []
    for snake in game_state["board"]["snakes"]:
        if snake["id"] != game_state["you"]["id"]:
            enemies.append((snake["body"][0]["x"], snake["body"][0]["y"]))

    obs_vector = []
    feature_order = [
        "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
        "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
        "food_score", "food_delta", "is_food", "dist_to_center"
    ]
    
    action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    # Собираем 68 фичей точно так же, как при обучении
    for act_idx in range(4):
        move = action_map[act_idx]
        if move not in legal:
            obs_vector.extend([0.0] * 17)
            continue
            
        try:
            dx_dir, dy_dir = DIRECTIONS[move]
            nxt = (head[0] + dx_dir, head[1] + dy_dir)
            
            feats = _candidate_features(game_state, move)
            
            # 13 базовых фичей с нормализацией под 11х11
            for name in feature_order:
                val = float(feats.get(name, 0.0))
                if name in ["space_capped", "open_space", "voronoi"]:
                    val /= 121.0
                elif name in ["wall_dist", "dist_to_center"]:
                    val /= 11.0
                obs_vector.append(val)
                
            # 4 Пространственные фичи (векторы до еды и врага из будущей точки)
            food_dx, food_dy = 0.0, 0.0
            if foods:
                closest_f = min(foods, key=lambda p: abs(p[0]-nxt[0]) + abs(p[1]-nxt[1]))
                food_dx = (closest_f[0] - nxt[0]) / 11.0
                food_dy = (closest_f[1] - nxt[1]) / 11.0
                
            enemy_dx, enemy_dy = 0.0, 0.0
            if enemies:
                closest_e = min(enemies, key=lambda p: abs(p[0]-nxt[0]) + abs(p[1]-nxt[1]))
                enemy_dx = (closest_e[0] - nxt[0]) / 11.0
                enemy_dy = (closest_e[1] - nxt[1]) / 11.0
                
            obs_vector.extend([food_dx, food_dy, enemy_dx, enemy_dy])
            
        except Exception:
            obs_vector.extend([0.0] * 17)

    obs_array = np.array(obs_vector, dtype=np.float32)
    
    # Инференс модели PPO
    action, _states = RL_MODEL.predict(obs_array, deterministic=True)
    chosen_move = action_map[int(action)]
    
    # Железная страховка: если нейросеть ошиблась и выбрала смерть, берем первый безопасный ход
    if chosen_move not in legal:
        return legal[0]
        
    return chosen_move


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
