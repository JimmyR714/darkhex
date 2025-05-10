"""Some utility functions"""
import tkinter as tk
import logging
import os
import math
from scipy.cluster.hierarchy import DisjointSet

# maps letters to colors for coloring in board
colour_map = {
    "w": "white",
    "b": "black",
    "e": "grey"
}

def swap_colour(colour : str) -> str:
    """
    Returns the opposite colour to what has been input
    """
    match colour:
        case "b":
            return "w"
        case "w":
            return "b"
        case _:
            raise ValueError("Invalid colour given to _swap_colour")


def draw_hex(centre: tuple[int, int], outer_size : int,
         canvas : tk.Canvas, border_colour="black", bkg_colour="") -> int:
    """Function to draw a hexagon"""
    def vertex(i):
        """Returns one vertex of the hexagon"""
        angle_deg = (60 * i) - 30
        angle_rad = (math.pi / 180) * angle_deg
        return (centre[0] + outer_size * math.cos(angle_rad),
            centre[1] + outer_size * math.sin(angle_rad)
        )
    logging.debug("Drawing hex with centre %s, outer size %s", centre, outer_size)
    # collect coords for the hexagon
    coords = []
    for i in range(6):
        vertex_i = vertex(i)
        coords.append(vertex_i[0])
        coords.append(vertex_i[1])
    coords.append(coords[0])
    coords.append(coords[1])
    canvas_id = canvas.create_polygon(coords, outline=border_colour, fill=bkg_colour, width=1)
    logging.debug("Hex drawn with coords %s", coords)
    return canvas_id


def create_default_components(num_cols: int, num_rows: int) -> tuple[DisjointSet, DisjointSet]:
    """
    Creates the default connected components for a board of size num_cols x num_rows. 
    Returns (white_components, black_components)
    """
    # set starting components
    white_components = DisjointSet([])
    black_components = DisjointSet([])
    # initial white components are left and right columns
    for y in range(num_rows+2):
        white_components.add((0,y))
        white_components.merge((0,0), (0,y))
        white_components.add((num_cols+1, y))
        white_components.merge((num_cols+1,0), (num_cols+1, y))

    # initial black components are top and bottom rows
    for x in range(1,num_cols+1):
        black_components.add((x, 0))
        black_components.merge((1,0), (x,0))
        black_components.add((x, num_rows+1))
        black_components.merge((1, num_rows+1), (x, num_rows+1))

    return (white_components, black_components)


def update_components(cell_pos: tuple[int, int], board: list[list[str]],
                      components: tuple[DisjointSet, DisjointSet], colour : str,
                      borders = True) -> None:
    """
    Update the connected components of the given colour to include the new cell (col,row).
    Col=1, row=1 is the top left placable cell
    
    """
    col = cell_pos[0]
    row = cell_pos[1]
    if borders:
        full_board = board
    else:
        full_board = add_borders(board)
    logging.debug("Attempting to merge components surrounding (%s, %s)", col, row)
    match colour:
        case "w":
            chosen_components = components[0]
        case "b":
            chosen_components = components[1]
        case _:
            raise ValueError("Invalid colour given to update_components")

    chosen_components.add((col,row))
    # attempt to connect to each matching colour in surrounding hex
    adj = [
        (col-1, row), (col, row-1), (col+1, row-1),
        (col+1, row), (col, row+1), (col-1,row+1)
    ]
    for cell in adj:
        # if adjacent cell is of the same colour
        if colour in full_board[cell[1]][cell[0]] and (cell[0], cell[1]) in chosen_components:
            # connect the components
            chosen_components.merge((col,row),cell)
            logging.debug("(%s, %s) and %s %s components merged",
                            col, row, cell, colour_map[colour]
            )


def win_check(num_rows: int, num_cols: int, white_components: DisjointSet,
              black_components: DisjointSet) -> str:
    """
    Check if the board is in a winning position for some player.
    Returns:
        "black_win" if black has won
        "white_win" if white has won
        "none" if nobody has won
    """
    #check if the two black rows are connected or the two white columns are connected
    logging.info("Performing a win check")
    if black_components.connected((1, 0), (1, num_rows+1)):
        return "black_win"
    elif white_components.connected((0, 1), (num_cols+1, 1)):
        return "white_win"
    else:
        return "none"


def add_borders(board: list[list[str]]) -> list[list[str]]:
    """
    Add borders to a board with none
    """
    num_rows = len(board)
    num_cols = len(board[0])
    row = ["w"] + ["b" for i in range(num_cols)] + ["w"]
    return [row] + [
        ["w"] + board[j] + ["w"] for j in range(num_rows)
    ] + [row]


def create_board(num_cols: int, num_rows: int) -> list[list[str]]:
    """
    Create a starting board of size cols+1 x rows+1
    The top and bottom rows are black, the left and right columns are white
    """
    row = ["w"] + ["b" for i in range(num_cols)] + ["w"]
    return [row] + [
        ["w"] + ["e" for i in range(num_cols)] + ["w"] for j in range(num_rows)
    ] + [row]


def utility(util_config: dict, pos_info: dict) -> float:
    """
    Calculates the utility of a position according to the config and position info.
    Parameters:
        util_config: dict - Config of which parameters to evaluate and by what degree.
        pos_info: dict - Relevant information about the position
    Returns:
        util_score: float - Positive values represent a good position for white,
            negative values represent a good position for black.
    """
    def component_strength(components: DisjointSet) -> float:
        """
        Calculates strength of the connected components
        """
        total = 1.0
        #for now, just based on size of components
        for component in components:
            total += len(component)
        return total


    def seen_strength(colour: str) -> float:
        """
        Calculates strength of the seen elements on the board for colour
        """
        total = 1.0
        for row in board:
            for cell in row:
                if "u" in cell and colour in cell:
                    #if they haven't seen our cell
                    total += 3
                elif "s" in cell and swap_colour(colour) in cell:
                    #if we have seen their cell
                    total += 5
                elif "u" in cell and swap_colour(colour) in cell:
                    #we haven't seen their cell
                    total -= 3
                elif "s" in cell and colour in cell:
                    #they have seen our cell
                    total -= 5
        return total


    def cell_exists(cell) -> bool:
        """
        Check whether a potential cell exists
        
        Parameters:
            cell: tuple[int, int] - The cell that we are checking the existence of.
        """
        top_left = cell[0] >= 0 and cell[1] >= 0
        bottom_right = cell[0] < num_cols and cell[1] < num_rows
        return top_left and bottom_right


    def add_cond(cond: str,  cells: list[tuple[int, int]], colour: str):
        """
        Calculates the value arising from the condition within the cells provided.
        
        Parameters:
            colour: str - The colour that we are checking for
            cond: str - The condition we will be adding.
            cells: list[tuple[int, int]] - The cells that the condition includes
            
        Returns:
            cond_weightings : int - The sum of weightings from this condition.
        """
        cond_weighting = 0
        # for each cell valid in this condition
        for cell in cells:
            #check if the cell is real and the correct colour
            if cell_exists(cell) and board[cell[0]][cell[1]] == colour:
                # add the cell to the function
                cond_weighting += util_config[cond]
        return cond_weighting


    #get the board from the config
    assert "board" in pos_info
    board = pos_info["board"]
    num_rows = len(board)
    num_cols = len(board[0])

    if "win" in util_config:
        #start with win checks
        #check components are in info
        assert "components" in pos_info
        white_components = pos_info["components"]["w"]
        black_components = pos_info["components"]["b"]

        win = win_check(
            num_rows=len(board),
            num_cols=len(board[0]),
            white_components=white_components,
            black_components=black_components
        )
        if win == "white_win":
            return util_config["win"]
        elif win == "black_win":
            return -1 * util_config["win"]

    #perform an ordinary evaluation
    total_utility = 0.0

    #check for connected component strength
    if "connected" in util_config:
        #check components are in pos_info
        assert "components" in pos_info
        white_components = pos_info["components"]["w"]
        black_components = pos_info["components"]["b"]

        white_component_strength = component_strength(white_components)
        black_component_strength = component_strength(black_components)
        total_utility += util_config["components"] * (
            white_component_strength - black_component_strength)

    #check for seen strength
    if "seen" in util_config:
        #then include seen strength
        total_utility += util_config["seen"] * (seen_strength("w") - seen_strength("b"))

    #check for vc and/or vsc simultaneously
    if "width_2_vc" in util_config or "width_2_semi_vc" in util_config or "width_1" in util_config:
        #collect the valid cells for each of the inputted conditions
        valid_cells = {}
        conditions = []
        if "width_1" in util_config:
            valid_cells.update({"width_1": lambda col, row: [
                (col-1, row), (col, row-1), (col+1, row-1),
                (col+1, row), (col, row+1), (col-1,row+1)
            ]})
            conditions.append("width_1")
        if "width_2_vc" in util_config:
            valid_cells.update({"width_2_vc": lambda col, row: [
                (col+1, row-2), (col+2, row-1), (col+1, row+1),
                (col-1, row+1), (col-2, row+1), (col-1, row-1)
            ]})
            conditions.append("width_2_vc")
        if "width_2_semi_vc" in util_config:
            valid_cells.update({"width_2_semi_vc": lambda col, row: [
                (col+2, row-1), (col+2, row), (col, row+2),
                (col-2, row+2), (col-2, row), (col, row-2)
            ]})
            conditions.append("width_2_semi_vc")
        #loop through valid cells, adding correct amount
        for row in range(num_rows):
            for col in range(num_cols):
                for cond in conditions:
                    total_utility += add_cond(cond, valid_cells[cond](col, row), "w")
                    total_utility -= add_cond(cond, valid_cells[cond](col, row), "b")

    return total_utility
