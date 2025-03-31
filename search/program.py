# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers

from .core import CellState, Coord, Direction, MoveAction
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
            coordinates to "player cours". The keys are `Coord` instances,
            and the values are `CellState` instances which can be one of
            `CellState.RED`, `CellState.BLUE`, or `CellState.LILY_PAD`.
    
    Returns:
        A list of "move actions" as MoveAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a cour-coded version!
    print(render_board(board, ansi=True))

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...
    STATE = 0
    PARENT = 1
    ACTION = 2
    DEPTH = 3
    CHILDREN = 4

    actions = ["DOWN", "LEFT", "RIGHT", "DOWNLEFT", "DOWNRIGHT", "JUMPMOVE"] 
    initial_node = [board, None, None, 0, []]
    queue = [initial_node]
    visited_state = []
    while queue:
        node = queue.pop(0)
        if node[STATE] not in visited_state:
            visited_state.append(node[STATE])
            if node[DEPTH] < 5:
                for action in actions:
                    new_state = node[STATE].copy()
                    new_state = take_action(new_state, action)
                    if new_state:
                        new_node = [new_state, node, action, node[DEPTH] + 1, []]
                        node[CHILDREN].append(new_node)
                        queue.append(new_node)
    
def take_action(state, action):
    BOARD_SIZE = 8  # Define the size of the board

    def is_within_bounds(coord):
        """Check if a coordinate is within the bounds of the board."""
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE
    
    def get_destination_coord(from_coord, over_coord):
        """Calculate the destination coordinate for a jump move."""
        delta_r = over_coord.r - from_coord.r
        delta_c = over_coord.c - from_coord.c
        return Coord(over_coord.r + delta_r, over_coord.c + delta_c)
    

    for coordinate in state:
        if state[coordinate] == CellState.RED:
            red_coord = coordinate
            break
    
    if action == "DOWN" and 0 <= red_coord.r < 7 and state[red_coord.r + 1, red_coord.c] == CellState.LILY_PAD:
        state[red_coord.r + 1, red_coord.c] = CellState.RED
        del state[red_coord]
        return state

    if action == "LEFT" and 0 < red_coord.c <= 7 and state[red_coord.r, red_coord.c - 1] == CellState.LILY_PAD:
        state[red_coord.r, red_coord.c - 1] = CellState.RED
        del state[red_coord]
        return state
    
    if action == "RIGHT" and 0 <= red_coord.c < 7 and state[red_coord.r, red_coord.c + 1] == CellState.LILY_PAD:
        state[red_coord.r, red_coord.c + 1] = CellState.RED
        del state[red_coord]
        return state
    
    if action == "DOWNLEFT" and 0 <= red_coord.r < 7 and 0 < red_coord.c <= 7 and state[red_coord.r + 1, red_coord.c - 1] == CellState.LILY_PAD:
        state[red_coord.r - 1, red_coord.c - 1] = CellState.RED
        del state[red_coord]
        return state
    
    if action == "DOWNRIGHT" and 0 <= red_coord.r < 7 and 0 <= red_coord.c < 7 and state[red_coord.r + 1, red_coord.c + 1] == CellState.LILY_PAD:
        state[red_coord.r + 1, red_coord.c + 1] = CellState.RED
        del state[red_coord]
        return state
    
    if action == "JUMPMOVE":
            # Possible adjacent positions to check for jumps
            adjacent_coords = [
                Coord(red_coord.r + 1, red_coord.c),      # DOWN
                Coord(red_coord.r, red_coord.c + 1),      # RIGHT
                Coord(red_coord.r, red_coord.c - 1),      # LEFT
                Coord(red_coord.r + 1, red_coord.c + 1),  # DOWNRIGHT
                Coord(red_coord.r + 1, red_coord.c - 1)   # DOWNLEFT
            ]
            
            for coord in adjacent_coords:
                if not is_within_bounds(coord):
                    continue  # Skip if the coordinate is out of bounds
                
                if state[coord] == CellState.BLUE:
                    # Calculate where the frog would land if it jumped over the Blue frog
                    destination = get_destination_coord(red_coord, coord)
                    
                    if is_within_bounds(destination) and state[destination] == CellState.LILY_PAD:
                        # Make the jump move
                        state[destination] = CellState.RED
                        del state[red_coord]    # Remove the old Red frog position
                        return state  # Return the updated state

    return None

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
