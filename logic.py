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

import time
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
        "color": "#FF6B1A",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using lookahead search, falling back to the
    1-ply model, then the heuristic, so gameplay always returns a legal move.
    """
    move = None
    try:
        move = choose_move_search(game_state)
    except Exception:  # noqa: BLE001 - search must never break gameplay
        move = None
    if move is None:
        try:
            move = choose_move_model(game_state)
        except Exception:  # noqa: BLE001 - a model issue must never break gameplay
            move = None
    if move is None:
        move = choose_move_heuristic(game_state)
    return _avoid_avoidable_h2h(game_state, move)


def _avoid_avoidable_h2h(game_state: Dict, move: str) -> str:
    """Override the chosen move if it risks a losing/tying head-to-head that a
    safer legal move would sidestep without trapping us.

    The model and heuristic only *penalize* head-to-head risk in their score,
    so a risky move can still win if other terms outweigh it. This is a hard
    safety net on top of that scoring.
    """
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]

    legal = _legal_moves(game_state)
    if move not in legal or len(legal) <= 1:
        return move

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)

    dx, dy = DIRECTIONS[move]
    if (head[0] + dx, head[1] + dy) not in danger:
        return move

    best_alt, best_space = None, -1
    for alt in legal:
        adx, ady = DIRECTIONS[alt]
        anxt = (head[0] + adx, head[1] + ady)
        if anxt in danger:
            continue
        space = _flood_fill(anxt, occupied, width, height, limit=my_length + 1)
        if space >= my_length + 1 and space > best_space:
            best_alt, best_space = alt, space

    return best_alt or move


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
    behind_in_length = my_length <= _max_enemy_length(board["snakes"], you["id"])

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

        # When hungry, or shorter than the biggest opponent, nudge toward food
        # — losing a length race means losing every future head-to-head.
        if foods and (health < HUNGRY_THRESHOLD or behind_in_length):
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


def _max_enemy_length(snakes: List[Dict], my_id: str) -> int:
    lengths = [s["length"] for s in snakes if s["id"] != my_id]
    return max(lengths) if lengths else 0


# --- Lookahead search ---------------------------------------------------------
#
# Paranoid iterative-deepening minimax with alpha-beta pruning. We model
# ourselves (MAX) against only the single nearest opponent (MIN); every other
# snake on the board is treated as a static obstacle for the duration of the
# search. This keeps the branching factor at a fixed 4x4 per simulated turn
# regardless of how many snakes are in the game, which is what makes a few
# plies of search tractable in Python within the move timeout.
#
# Iterative deepening means we always have a legal answer: we search depth 1,
# then 2, then 3, ... and keep the best move from the deepest search that
# finished before the time budget ran out. A deeper search in progress is
# simply abandoned (via ``_TimeUp``) rather than returned partially.

_TIME_BUDGET_SECONDS = 0.35
_MAX_SEARCH_DEPTH = 8
_LOSS_SCORE = -1_000_000.0
_WIN_SCORE = 1_000_000.0


class _TimeUp(Exception):
    """Raised to unwind the search once the time budget is exhausted."""


def _find_snake(board: Dict, snake_id: str) -> Optional[Dict]:
    for snake in board["snakes"]:
        if snake["id"] == snake_id:
            return snake
    return None


def _nearest_opponent_id(board: Dict, my_id: str) -> Optional[str]:
    you = _find_snake(board, my_id)
    if you is None:
        return None
    head = (you["head"]["x"], you["head"]["y"])
    opponents = [s for s in board["snakes"] if s["id"] != my_id]
    if not opponents:
        return None
    opponents.sort(key=lambda s: _manhattan(head, (s["head"]["x"], s["head"]["y"])))
    return opponents[0]["id"]


def _legal_moves_for(board: Dict, snake_id: str) -> List[str]:
    snake = _find_snake(board, snake_id)
    if snake is None:
        return []
    width, height = board["width"], board["height"]
    head = (snake["head"]["x"], snake["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]


def _apply_turn(board: Dict, moves: Dict[str, str]) -> Dict:
    """Simulate one full game turn for the snakes named in ``moves``.

    Snakes not in ``moves`` are left completely untouched (frozen obstacles).
    Implements move/grow/starve, then eliminates snakes that hit a wall, a
    body, or lose/tie a head-to-head collision, per the Battlesnake rules.
    """
    width, height = board["width"], board["height"]
    food_cells = {(f["x"], f["y"]) for f in board["food"]}

    proposals: Dict[str, Dict] = {}
    for snake in board["snakes"]:
        sid = snake["id"]
        if sid not in moves:
            continue
        dx, dy = DIRECTIONS[moves[sid]]
        head = (snake["head"]["x"], snake["head"]["y"])
        new_head = (head[0] + dx, head[1] + dy)
        body_pts = [(seg["x"], seg["y"]) for seg in snake["body"]]
        ate = new_head in food_cells
        new_body = [new_head] + (body_pts if ate else body_pts[:-1])
        proposals[sid] = {
            "head": new_head,
            "body": new_body,
            "health": 100 if ate else snake["health"] - 1,
            "ate": ate,
        }

    all_bodies: Dict[str, List[Point]] = {}
    for snake in board["snakes"]:
        sid = snake["id"]
        all_bodies[sid] = proposals[sid]["body"] if sid in proposals else [
            (seg["x"], seg["y"]) for seg in snake["body"]
        ]

    dead: Set[str] = set()
    for sid, proposal in proposals.items():
        head = proposal["head"]
        if proposal["health"] <= 0:
            dead.add(sid)
            continue
        if not _in_bounds(head, width, height):
            dead.add(sid)
            continue
        if head in all_bodies[sid][1:]:
            dead.add(sid)
            continue
        collided = False
        for other_sid, other_body in all_bodies.items():
            if other_sid == sid:
                continue
            if head == other_body[0]:
                # Head-to-head: the shorter (or equal-length) snake dies. A
                # frozen opponent's head is just an obstacle it never moved
                # away from, so running into it is always fatal.
                if other_sid not in proposals or len(all_bodies[sid]) <= len(other_body):
                    collided = True
                    break
            elif head in other_body[1:]:
                collided = True
                break
        if collided:
            dead.add(sid)

    new_snakes: List[Dict] = []
    eaten_food: Set[Point] = set()
    for snake in board["snakes"]:
        sid = snake["id"]
        if sid not in proposals:
            new_snakes.append(snake)
            continue
        if sid in dead:
            continue
        proposal = proposals[sid]
        if proposal["ate"]:
            eaten_food.add(proposal["head"])
        new_snakes.append(
            {
                "id": sid,
                "health": proposal["health"],
                "length": len(proposal["body"]),
                "head": {"x": proposal["head"][0], "y": proposal["head"][1]},
                "body": [{"x": x, "y": y} for x, y in proposal["body"]],
            }
        )

    new_food = [f for f in board["food"] if (f["x"], f["y"]) not in eaten_food]
    return {"width": width, "height": height, "food": new_food, "snakes": new_snakes}


def _position_features(
    board: Dict, my_id: str, my_length: int, health: int, my_body: List[Dict], at: Point
) -> Dict[str, float]:
    """Feature vector for our snake occupying head position ``at``."""
    width, height = board["width"], board["height"]
    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], my_id, my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != my_id]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    my_dist = _bfs_dist([at], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    my_tail = (my_body[-1]["x"], my_body[-1]["y"])
    reach = _bfs_dist([at], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((at[0] + ddx, at[1] + ddy), width, height)
        and (at[0] + ddx, at[1] + ddy) not in occupied
    )

    behind_in_length = my_length <= _max_enemy_length(board["snakes"], my_id)
    nearest_dist = min((_manhattan(at, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD or behind_in_length

    return {
        "space_capped": float(_flood_fill(at, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(at, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if at in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(at, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(at, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(at[0], width - 1 - at[0], at[1], height - 1 - at[1])),
        "food_score": float((width + height - nearest_dist) * 2) if hungry and foods else 0.0,
        "is_food": 1.0 if at in foods else 0.0,
        "dist_to_center": abs(at[0] - (width - 1) / 2) + abs(at[1] - (height - 1) / 2),
        "_nearest_food_dist": float(nearest_dist),
    }


def _score_features(feats: Dict[str, float]) -> float:
    """Score a feature vector with the embedded standardized linear model."""
    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    score = _MODEL["intercept"]
    for i, name in enumerate(names):
        z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
        score += coef[i] * z
    return score


def _evaluate_board(board: Dict, my_id: str) -> float:
    """Static evaluation of a (possibly simulated) board from our perspective."""
    you = _find_snake(board, my_id)
    if you is None or you["health"] <= 0:
        return _LOSS_SCORE
    head = (you["head"]["x"], you["head"]["y"])
    feats = _position_features(board, my_id, you["length"], you["health"], you["body"], head)
    return _score_features(feats)


def _order_moves_quick(board: Dict, my_id: str, moves: List[str]) -> List[str]:
    """Cheap move ordering (by resulting open space) to help alpha-beta prune."""
    you = _find_snake(board, my_id)
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    occupied = _occupied_cells(board["snakes"])

    def key(move: str) -> int:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        return -_flood_fill(nxt, occupied, width, height, limit=my_length + 1)

    return sorted(moves, key=key)


def _max_node(
    board: Dict, my_id: str, opp_id: Optional[str], depth: int, alpha: float, beta: float, deadline: float
) -> float:
    if time.monotonic() > deadline:
        raise _TimeUp()
    if _find_snake(board, my_id) is None:
        return _LOSS_SCORE - depth
    if opp_id is not None and _find_snake(board, opp_id) is None:
        return _WIN_SCORE + depth
    if depth <= 0:
        return _evaluate_board(board, my_id)

    moves = _legal_moves_for(board, my_id)
    if not moves:
        return _LOSS_SCORE - depth

    best = float("-inf")
    for move in moves:
        val = _min_node(board, my_id, opp_id, move, depth, alpha, beta, deadline)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _min_node(
    board: Dict,
    my_id: str,
    opp_id: Optional[str],
    my_move: str,
    depth: int,
    alpha: float,
    beta: float,
    deadline: float,
) -> float:
    if time.monotonic() > deadline:
        raise _TimeUp()
    if opp_id is None or _find_snake(board, opp_id) is None:
        new_board = _apply_turn(board, {my_id: my_move})
        return _max_node(new_board, my_id, opp_id, depth - 1, alpha, beta, deadline)

    opp_moves = _legal_moves_for(board, opp_id) or ["up"]
    best = float("inf")
    for opp_move in opp_moves:
        new_board = _apply_turn(board, {my_id: my_move, opp_id: opp_move})
        val = _max_node(new_board, my_id, opp_id, depth - 1, alpha, beta, deadline)
        if val < best:
            best = val
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return best


def _search_root(
    board: Dict, my_id: str, opp_id: Optional[str], depth: int, deadline: float
) -> Tuple[Optional[str], float]:
    moves = _legal_moves_for(board, my_id)
    if not moves:
        return None, _LOSS_SCORE
    moves = _order_moves_quick(board, my_id, moves)

    alpha, beta = float("-inf"), float("inf")
    best_move, best_val = moves[0], float("-inf")
    for move in moves:
        val = _min_node(board, my_id, opp_id, move, depth, alpha, beta, deadline)
        if val > best_val:
            best_val, best_move = val, move
        if best_val > alpha:
            alpha = best_val
    return best_move, best_val


def choose_move_search(game_state: Dict) -> Optional[str]:
    """Iterative-deepening minimax/alpha-beta search against the nearest
    opponent, time-boxed to ``_TIME_BUDGET_SECONDS``. Returns the best move
    found at the deepest fully-completed depth, or ``None`` if not even a
    depth-1 search could finish (so the caller falls back to the 1-ply model).
    """
    board = game_state["board"]
    my_id = game_state["you"]["id"]
    if _find_snake(board, my_id) is None:
        return None
    opp_id = _nearest_opponent_id(board, my_id)

    deadline = time.monotonic() + _TIME_BUDGET_SECONDS
    best_move: Optional[str] = None
    depth = 1
    try:
        while depth <= _MAX_SEARCH_DEPTH:
            move, _ = _search_root(board, my_id, opp_id, depth, deadline)
            if move is None:
                break
            best_move = move
            depth += 1
    except _TimeUp:
        pass
    return best_move


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
    head = (you["head"]["x"], you["head"]["y"])

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    feats = _position_features(board, you["id"], you["length"], you["health"], you["body"], nxt)
    nearest_now = min(
        (_manhattan(head, (f["x"], f["y"])) for f in board["food"]), default=_BIG
    )
    feats["food_delta"] = float(nearest_now - feats["_nearest_food_dist"]) if board["food"] else 0.0
    return feats


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
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    best_move, best_score = None, float("-inf")
    for move in legal:
        score = _score_features(_candidate_features(game_state, move))
        if score > best_score:
            best_score, best_move = score, move
    return best_move


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
