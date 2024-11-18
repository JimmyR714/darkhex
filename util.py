"""Some utility functions"""
import tkinter as tk
import logging
import math

# maps letters to colours for colouring in board
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
    id = canvas.create_polygon(coords, outline=border_colour, fill=bkg_colour, width=1)
    logging.debug("Hex drawn with coords %s", coords)
    return id
