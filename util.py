"""Some utility functions"""

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