"""Some utility functions"""
import tkinter as tk
import logging
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
        #TODO makeshift bug fix
        if colour in full_board[cell[1]][cell[0]] and (cell[1], cell[0]) in chosen_components:
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
