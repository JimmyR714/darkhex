"""Module containing the general agent class"""
import random

class Agent():
    """
    General class for an agent that can play dark hex
    """
    def __init__(self, num_cols: int, num_rows: int, settings: dict):
        """
        Creates an agent that has a strategy for the correct size board.
        This creates the self stored board and remembers board size.
        
        Parameters:
            num_cols: int: The number of columns on the board (i..e the width). 
                Doesn't include borders.
            num_rows: int: The number of rows on the board (i..e the height). 
                Doesn't include borders.
            settings: dict: The settings for this agent, e.g. agent colour
        """
        self.colour = settings["colour"]
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board = [["e"] * num_cols] * num_rows


    def move(self) -> tuple[int, int]:
        """
        Make a move on the board.
        Uses some algorithm to decide upon the best move, and returns it.
        Doesn't add it to it's own board yet, since that cell may be full.
        
        Returns:
            (col, row): tuple[int, int]: the chosen cell to play in. 
                col = 1 is the left column.
                row = 1 is the top row.
        """
        #default move by general agent is just random
        return (random.randint(1, self.num_cols), random.randint(1, self.num_rows))


    def update_information(self, col: int, row: int, colour: str) -> bool:
        """
        Recieve some updated information about the board.
        The controller of this agent will call this after we make a move.
        Necessary to find out whether our move was successful.
        
        Parameters:
            col: int: The chosen column to play in.
                col = 1 is the left column.
            row: int: The chosen row to play in.
                row = 1 is the top row.
            colour: str: The colour of the new cell. "w" for white and "b" for black.
            
        Returns:
            True: When another move must be made.
            False: When no other move must be made.
        """
        self.board[row-1][col-1] = colour
        #i.e. if we discovered one of their pieces
        return self.colour != colour
