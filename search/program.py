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
     This is the entry point for your submission. You should modify this
     function to solve the search problem discussed in the Part A specification.
     See `core.py` for information on the types being used here.
 
     Parameters:
         `board`: a dictionary representing the initial board state, mapping
             coordinates to "player colours". The keys are `Coord` instances,
             and the values are `CellState` instances which can be one of
             `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
     
     Returns:
         A list of "move actions" as MoveAction instances, or `None` if no
         solution is possible.
     """
 
     # The render_board() function is handy for debugging. It will print out a
     # board state in a human-readable format. If your terminal supports ANSI
     # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, ansi=True))
 
     # Do some impressive AI stuff here to find the solution...
     # ...
     # ... (your solution goes here!)
     # ...



    # set initial variables 
    red_coord = None
    blue_set = set()
    lily_set = set()
    
    # extract the initial positions of red, blue and lily pads
    for coord, state in board.items():
        if state == CellState.RED:
            red_coord = coord
        elif state == CellState.BLUE:
            blue_set.add(coord)
        elif state == CellState.LILY_PAD:
            lily_set.add(coord)
    
    if red_coord is None:
        return None 
    
    # set initial state
    initial_state = (red_coord, frozenset(lily_set))
    
    # all the allowed directions for the red frog to move
    allowed_dirs = [
        Direction.Right, 
        Direction.Left, 
        Direction.Down, 
        Direction.DownLeft, 
        Direction.DownRight
    ]
    
    # help function to get the neighbor cell in a given direction
    # if the neighbor cell is out of bounds, return None
    def get_neighbor(coord: Coord, d: Direction, steps: int = 1) -> Coord | None:
        new_r = coord.r + d.value.r * steps
        new_c = coord.c + d.value.c * steps
        if 0 <= new_r < BOARD_N and 0 <= new_c < BOARD_N:
            return Coord(new_r, new_c)
        return None

    # generate all possible jump moves from the current position
    # return a list of tuples, each containing:
    # - landing position
    # - the path taken (list of directions)
    # - the landing positions (list of coordinates)
    def generate_jump_moves(red: Coord, current_lily: set[Coord]) -> list[tuple[Coord, list[Direction], list[Coord]]]:
        jump_moves = []
        # use a depth-first search (DFS) to find all possible jump sequences
        def dfs(current: Coord, path: list[Direction], landing_list: list[Coord], visited: set[Coord]):
            for d in allowed_dirs:
                # avoid going back in the same direction
                if path:
                    last = path[-1]
                    if (last == Direction.Right and d == Direction.Left) or (last == Direction.Left and d == Direction.Right):
                        continue
                neighbor = get_neighbor(current, d, steps=1)
                #check for out of bounds
                if neighbor is None:
                    continue
                # check for blue frog
                if neighbor not in blue_set:
                    continue
                landing = get_neighbor(current, d, steps=2)
                if landing is None:
                    continue
                # landing position must be a lily pad
                if landing not in current_lily:
                    continue
                new_path = path + [d]
                new_landing_list = landing_list + [landing]
                # record the jump move in a list
                jump_moves.append((landing, new_path, new_landing_list))

                if landing not in visited:
                    visited.add(landing)
                    dfs(landing, new_path, new_landing_list, visited)
                    visited.remove(landing)
        dfs(red, [], [], {red})
        return jump_moves



    # generate all the legal moves from the current state, returning a list of tuples
    # each containing:
    # - the move action (MoveAction object)
    # - the new state (a tuple of the new red position and the new lily pad set)
    def generate_moves(state: tuple[Coord, frozenset[Coord]]) -> list[tuple[MoveAction, tuple[Coord, frozenset[Coord]]]]:
        red, lily_fs = state
        current_lily = set(lily_fs)  # unfrozenset to allow modification
        moves = []
        # simple move
        for d in allowed_dirs:
            nbr = get_neighbor(red, d, steps=1)
            if nbr is not None and nbr in current_lily:
                move_action = MoveAction(red, [d])
                new_lily = current_lily.copy()
                
                # Remove the lily pad the red frog is jumping from
                if red in new_lily:
                    new_lily.remove(red)

                new_state = (nbr, frozenset(new_lily))
                moves.append((move_action, new_state))
        # jump move
        jump_sequences = generate_jump_moves(red, current_lily)
        for landing, path, landing_positions in jump_sequences:
            move_action = MoveAction(red, path)
            new_lily = current_lily.copy()
            # get rid of the initial position of the red frog
            if red in new_lily:
                new_lily.remove(red)
            new_state = (landing, frozenset(new_lily))
            moves.append((move_action, new_state))
        return moves

    # heuristic function to estimate the cost to reach the goal state
    # here we say that the frog can move at most 7 rows in one move
    def heuristic(state: tuple[Coord, frozenset[Coord]]) -> int:
        red, _ = state
        remaining = 7 - red.r
        return math.ceil(remaining / 7)

    counter = itertools.count()

    def reconstruct_board(red: Coord, lily_set: frozenset[Coord]) -> dict[Coord, CellState]:
        updated_board = {}
        updated_board[red] = CellState.RED
        for blue in blue_set:
            updated_board[blue] = CellState.BLUE
        for lily in lily_set:
            if lily not in updated_board:  # Don't overwrite RED or BLUE
                updated_board[lily] = CellState.LILY_PAD
        return updated_board

    # A* search algorithm
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
        # Goal check
        if red.r == 7:
            return actions

        # generate all possible moves from the current state
        for move_action, new_state in generate_moves(state):
            new_actions = actions + [move_action]
            new_g = g + 1  # step cost is 1
            new_f = new_g + heuristic(new_state)
            if new_state not in visited_states:
                heapq.heappush(open_set, (new_f, next(counter), new_g, new_state, new_actions))



    return None  # no solution found

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
