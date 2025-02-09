"""Module containing the game class that allows dark hex to run with multiple board sizes"""

import logging
from scipy.cluster.hierarchy import DisjointSet
import game.util as util

class AbstractDarkHex:
    """
    Class that can run a game of Dark Hex. 
    
    Initialise with the dimensions and an empty board will be created. 
    
    By default, white moves first.
    
    Assume that position (1,1) is the top left cell, and (2,1) is the top row, 2nd column etc.
    
    We have black "goals" at the top and bottom, and white "goals" at the left and right.
    """

    def __init__(self, _cols : int, _rows : int, _first_turn="w"):
        self.cols = _cols
        self.rows = _rows
        self.board = []
        self.black_board = []
        self.white_board = []
        self.black_components = DisjointSet([])
        self.white_components = DisjointSet([])
        self.turn = _first_turn  # for now, default first turn is white's
        self.first_turn = _first_turn  # save in case of reset
        logging.info("Board parameters defined")
        self.reset_board()  # set starting state of board and components

    def move(self, row : int, col : int, colour : str) -> str:
        """
        Attempt to place a piece of a given colour into a given cell
        Parameters:
            colour: "b" or "w" representing black and white respectively
            row: The row to insert on, with 1 being at the top
            col: The column to insert on, with 1 being at the left
        Returns:
            "black_win" if the cell is placed and this wins the game for black
            "white_win" if the cell is placed and this wins the game for white
            "placed" if the cell is placed and the game continues
            "full_white" if the cell is occupied with a white tile
            "full_black" if the cell is occupied with a black tile
        """
        # check that this player is allowed to move
        assert colour == self.turn

        cell = self.board[row][col]
        if cell != "e":  # if the chosen cell is not empty
            logging.info("Non-empty cell %s at (%s, %s)", cell, col, row)
            # update our view, since we know where their piece is now
            self.get_board(colour)[row][col] = self.board[row][col]  # view update
            return "full_" + util.colour_map[self.board[row][col]]
        else:
            # update global board and our view
            self.board[row][col] = colour
            self.get_board(colour)[row][col] = colour
            self.turn = util.swap_colour(colour)  # swap turn
            util.update_components(
                cell_pos=(col, row),
                board=self.board,
                components=(self.white_components, self.black_components),
                colour=colour
            )  # update components
            logging.info("%s played at (%s, %s)",
                         util.colour_map[colour], col, row)
            win_check = util.win_check(
                self.rows,
                self.cols,
                self.white_components,
                self.black_components
            )
            logging.info("Result of win_check is: %s", win_check)
            if win_check != "none":
                return win_check
            else:
                return "placed"


    def reset_board(self) -> None:
        """
        Restart the game with the same board dimensions
        """
        # reset board contents
        self.board = util.create_board(num_cols=self.cols, num_rows=self.rows)
        self.black_board = util.create_board(num_cols=self.cols, num_rows=self.rows)
        self.white_board = util.create_board(num_cols=self.cols, num_rows=self.rows)

        # revert to correct first turn
        self.turn = self.first_turn

        (self.white_components, self.black_components) = util.create_default_components(
            self.cols,
            self.rows
        )

        logging.info("Board reset")


    def get_board(self, colour : str) -> list[list[str]]:
        """
        Returns the board for a given colour
        """
        match colour:
            case "b":
                return self.black_board
            case "w":
                return self.white_board
            case _:
                raise ValueError("Invalid colour given to get_board")
