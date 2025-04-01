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
            # 对于跳跃移动，移除沿途所有落脚的 lily pad
            for pos in landing_positions:
                if pos in new_lily:
                    new_lily.remove(pos)
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

# from .core import CellState, Coord, Direction, MoveAction
# from .utils import render_board


# def search(
#     board: dict[Coord, CellState]
# ) -> list[MoveAction] | None:
#     """
#     This is the entry point for your submission. You should modify this
#     function to solve the search problem discussed in the Part A specification.
#     See `core.py` for information on the types being used here.

#     Parameters:
#         `board`: a dictionary representing the initial board state, mapping
#             coordinates to "player cours". The keys are `Coord` instances,
#             and the values are `CellState` instances which can be one of
#             `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
#     Returns:
#         A list of "move actions" as MoveAction instances, or `None` if no
#         solution is possible.
#     """

#     # The render_board() function is handy for debugging. It will print out a
#     # board state in a human-readable format. If your terminal supports ANSI
#     # codes, set the `ansi` flag to True to print a cour-coded version!
#     print(render_board(board, ansi=True))

#     # Do some impressive AI stuff here to find the solution...
#     # ...
#     # ... (your solution goes here!)
#     # ...
#     STATE = 0
#     PARENT = 1
#     ACTION = 2
#     DEPTH = 3
#     CHILDREN = 4

#     actions = ["DOWN", "LEFT", "RIGHT", "DOWNLEFT", "DOWNRIGHT", "JUMPMOVE"] 
#     initial_node = [board, None, None, 0, []]
#     queue = [initial_node]
#     visited_state = []
#     while queue:
#         node = queue.pop(0)
#         if node[STATE] not in visited_state:
#             visited_state.append(node[STATE])
#             if node[DEPTH] < 5:
#                 for action in actions:
#                     new_state = node[STATE].copy()
#                     new_state = take_action(new_state, action)
#                     if new_state:
#                         new_node = [new_state, node, action, node[DEPTH] + 1, []]
#                         node[CHILDREN].append(new_node)
#                         queue.append(new_node)
    
# def take_action(state, action):
#     BOARD_SIZE = 8  # Define the size of the board

#     def is_within_bounds(coord):
#         """Check if a coordinate is within the bounds of the board."""
#         return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE
    
#     def get_destination_coord(from_coord, over_coord):
#         """Calculate the destination coordinate for a jump move."""
#         delta_r = over_coord.r - from_coord.r
#         delta_c = over_coord.c - from_coord.c
#         return Coord(over_coord.r + delta_r, over_coord.c + delta_c)
    

#     for coordinate in state:
#         if state[coordinate] == CellState.RED:
#             red_coord = coordinate
#             break
    
#     if action == "DOWN" and 0 <= red_coord.r < 7 and state[red_coord.r + 1, red_coord.c] == CellState.LILY_PAD:
#         state[red_coord.r + 1, red_coord.c] = CellState.RED
#         del state[red_coord]
#         return state

#     if action == "LEFT" and 0 < red_coord.c <= 7 and state[red_coord.r, red_coord.c - 1] == CellState.LILY_PAD:
#         state[red_coord.r, red_coord.c - 1] = CellState.RED
#         del state[red_coord]
#         return state
    
#     if action == "RIGHT" and 0 <= red_coord.c < 7 and state[red_coord.r, red_coord.c + 1] == CellState.LILY_PAD:
#         state[red_coord.r, red_coord.c + 1] = CellState.RED
#         del state[red_coord]
#         return state
    
#     if action == "DOWNLEFT" and 0 <= red_coord.r < 7 and 0 < red_coord.c <= 7 and state[red_coord.r + 1, red_coord.c - 1] == CellState.LILY_PAD:
#         state[red_coord.r - 1, red_coord.c - 1] = CellState.RED
#         del state[red_coord]
#         return state
    
#     if action == "DOWNRIGHT" and 0 <= red_coord.r < 7 and 0 <= red_coord.c < 7 and state[red_coord.r + 1, red_coord.c + 1] == CellState.LILY_PAD:
#         state[red_coord.r + 1, red_coord.c + 1] = CellState.RED
#         del state[red_coord]
#         return state
    
#     if action == "JUMPMOVE":
#             # Possible adjacent positions to check for jumps
#             adjacent_coords = [
#                 Coord(red_coord.r + 1, red_coord.c),      # DOWN
#                 Coord(red_coord.r, red_coord.c + 1),      # RIGHT
#                 Coord(red_coord.r, red_coord.c - 1),      # LEFT
#                 Coord(red_coord.r + 1, red_coord.c + 1),  # DOWNRIGHT
#                 Coord(red_coord.r + 1, red_coord.c - 1)   # DOWNLEFT
#             ]
            
#             for coord in adjacent_coords:
#                 if not is_within_bounds(coord):
#                     continue  # Skip if the coordinate is out of bounds
                
#                 if state[coord] == CellState.BLUE:
#                     # Calculate where the frog would land if it jumped over the Blue frog
#                     destination = get_destination_coord(red_coord, coord)
                    
#                     if is_within_bounds(destination) and state[destination] == CellState.LILY_PAD:
#                         # Make the jump move
#                         state[destination] = CellState.RED
#                         del state[red_coord]    # Remove the old Red frog position
#                         return state  # Return the updated state

#     return None

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
