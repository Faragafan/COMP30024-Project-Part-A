# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

import heapq
import math
import itertools
from .core import CellState, Coord, Direction, MoveAction, BOARD_N
from .utils import render_board

def search(
    board: dict[Coord, CellState]
) -> list[MoveAction] | None:
    """
    使用 A* 算法解决 Single Player Freckers 问题。
    状态由一个元组 (red_coord, lily_set) 表示，其中：
      - red_coord: 当前红青蛙的位置
      - lily_set: 一个 frozenset，包含所有未被占用的 lily pad 坐标
    蓝青蛙的位置在整个搜索过程中保持不变。
    
    合法的 MOVE 动作包括：
      - 简单移动：在红方允许的方向上一步移动到相邻的 lily pad（移动后原位置的 lily pad 消失）。
      - 跳跃移动：在允许的方向上，若相邻位置有青蛙（蓝青蛙）且落脚点（连续两步）有 lily pad，
        则可以进行跳跃。对于连续跳跃，沿途所有落脚的 lily pad 均被移除。
    
    每个 MOVE 动作的代价均为 1，启发函数采用：假设每次移动最多能前进 2 行，估计剩余步数。
    """
    # 打印棋盘，便于调试
    print(render_board(board, ansi=True))
    
    # 提取初始状态：寻找红青蛙、蓝青蛙以及所有 lily pad
    red_coord = None
    blue_set = set()
    lily_set = set()
    
    for coord, state in board.items():
        if state == CellState.RED:
            red_coord = coord
        elif state == CellState.BLUE:
            blue_set.add(coord)
        elif state == CellState.LILY_PAD:
            lily_set.add(coord)
    
    if red_coord is None:
        return None  # 未找到红青蛙
    
    # 状态表示为 (red_coord, frozenset(lily_set))
    initial_state = (red_coord, frozenset(lily_set))
    
    # 红方允许的移动方向
    allowed_dirs = [
        Direction.Right, 
        Direction.Left, 
        Direction.Down, 
        Direction.DownLeft, 
        Direction.DownRight
    ]
    
    # 辅助函数：获取非环绕的相邻坐标
    def get_neighbor(coord: Coord, d: Direction, steps: int = 1) -> Coord | None:
        new_r = coord.r + d.value.r * steps
        new_c = coord.c + d.value.c * steps
        if 0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N:
            return Coord(new_r, new_c)
        return None

    # 生成跳跃移动序列
    # 返回列表中每个元素为 (final_landing, path, landing_positions)
    # 其中 path 为方向序列，landing_positions 为跳跃过程中所有落脚的 lily pad 坐标（依次）
    def generate_jump_moves(red: Coord, current_lily: set[Coord]) -> list[tuple[Coord, list[Direction], list[Coord]]]:
        jump_moves = []
        # 使用 DFS 生成跳跃序列
        def dfs(current: Coord, path: list[Direction], landing_list: list[Coord], visited: set[Coord]):
            for d in allowed_dirs:
                if path:
                    last = path[-1]
                    if (last == Direction.Right and d == Direction.Left) or (last == Direction.Left and d == Direction.Right):
                        continue
                neighbor = get_neighbor(current, d, steps=1)
                if neighbor is None:
                    continue
                # 跳跃要求相邻单元格中必须有青蛙（这里只考虑蓝青蛙，因为红青蛙即将移动）
                if neighbor not in blue_set:
                    continue
                landing = get_neighbor(current, d, steps=2)
                if landing is None:
                    continue
                # 落脚点必须有未被占用的 lily pad
                if landing not in current_lily:
                    continue
                new_path = path + [d]
                new_landing_list = landing_list + [landing]
                # 记录该跳跃序列
                jump_moves.append((landing, new_path, new_landing_list))
                # 防止在同一跳跃序列中循环
                if landing not in visited:
                    visited.add(landing)
                    dfs(landing, new_path, new_landing_list, visited)
                    visited.remove(landing)
        dfs(red, [], [], {red})
        return jump_moves

    # 生成当前状态下所有合法移动（返回每个移动对应的 MoveAction 和更新后的状态）
    def generate_moves(state: tuple[Coord, frozenset[Coord]]) -> list[tuple[MoveAction, tuple[Coord, frozenset[Coord]]]]:
        red, lily_fs = state
        current_lily = set(lily_fs)  # 使用可变集合副本进行操作
        moves = []
        # 简单移动：一步移动
        for d in allowed_dirs:
            nbr = get_neighbor(red, d, steps=1)
            if nbr is not None and nbr in current_lily:
                move_action = MoveAction(red, [d])
                new_lily = current_lily.copy()
                # 移除目的地 lily pad（青蛙跳离后，该 lily pad 被占用，因此消失）
                new_lily.remove(nbr)
                new_state = (nbr, frozenset(new_lily))
                moves.append((move_action, new_state))
        # 跳跃移动（可能为多跳）
        jump_sequences = generate_jump_moves(red, current_lily)
        for landing, path, landing_positions in jump_sequences:
            move_action = MoveAction(red, path)
            new_lily = current_lily.copy()
            # 对于跳跃移动，只移除起始位置的 lily pad
            if red in new_lily:
                new_lily.remove(red)
            new_state = (landing, frozenset(new_lily))
            moves.append((move_action, new_state))
        return moves

    # 启发函数：假设每次移动最多能前进 2 行，估计剩余步数
    def heuristic(state: tuple[Coord, frozenset[Coord]]) -> int:
        red, _ = state
        remaining = 7 - red.r
        return math.ceil(remaining / 7)

    counter = itertools.count()

    # A* 搜索
    open_set = []
    # Each heap element: (f, count, g, state, actions)
    heapq.heappush(open_set, (heuristic(initial_state), next(counter), 0, initial_state, []))
    visited_states = set()
    
    while open_set:
        f, _, g, state, actions = heapq.heappop(open_set)
        if state in visited_states:
            continue
        visited_states.add(state)
        red, _ = state
        #目标测试：红青蛙是否已到达最后一行
        if red.r == 7:
            return actions
        #扩展邻居状态
        for move_action, new_state in generate_moves(state):
            new_actions = actions + [move_action]
            new_g = g + 1  # 每次移动代价为 1
            new_f = new_g + heuristic(new_state)
            if new_state not in visited_states:
                heapq.heappush(open_set, (new_f, next(counter), new_g, new_state, new_actions))
    
    return None  # 无解


    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
    # return [
    #     MoveAction(Coord(0, 5), [Direction.Down]),
    #     MoveAction(Coord(1, 5), [Direction.DownLeft]),
    #     MoveAction(Coord(3, 3), [Direction.Left]),
    #     MoveAction(Coord(3, 2), [Direction.Down, Direction.Right]),
    #     MoveAction(Coord(5, 4), [Direction.Down]),
    #     MoveAction(Coord(6, 4), [Direction.Down]),
    # ]
