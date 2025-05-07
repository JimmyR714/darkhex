"""Module containing the game class that allows hex to run with multiple board sizes"""

import logging
from scipy.cluster.hierarchy import DisjointSet
import game.util as util

class AbstractHex:
    """
    Class that can run a game of Hex. 
    
    Initialise with the dimensions and an empty board will be created. 
    
    By default, white moves first.
    
    Assume that position (1,1) is the top left cell, and (2,1) is the top row, 2nd column etc.
    
    We have black "goals" at the top and bottom, and white "goals" at the left and right.
    """

    def __init__(self, cols : int, rows : int):
        self.cols = cols
        self.rows = rows
        self.board = []
        self.black_components = DisjointSet([])
        self.white_components = DisjointSet([])
        self.turn = "w"  # for now, default first turn is white's
        logging.info("Board parameters defined")
        self.reset_board()  # set starting state of board and components


    def move(self, row : int, col : int, colour : str, fast = False) -> str:
        """
        Place a piece of a given colour into a given cell
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
        if fast:
            self.board[row][col] = colour
            util.update_components(
                cell_pos=(col, row),
                board=self.board,
                components=(self.white_components, self.black_components),
                colour=colour
            )  # update components
            return ""
        else:
            cell = self.board[row][col]
            if cell != "e":  # if the chosen cell is not empty
                return "full_" + util.colour_map[self.board[row][col]]
            else:
                self.board[row][col] = colour
                self.turn = util.swap_colour(colour)  # swap turn
                util.update_components(
                    cell_pos=(col, row),
                    board=self.board,
                    components=(self.white_components, self.black_components),
                    colour=colour
                )  # update components
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

        # revert to correct first turn
        self.turn = "w"

        (self.white_components, self.black_components) = util.create_default_components(
            self.cols,
            self.rows
        )

        logging.info("Board reset")


    def set_board(self, board: list[int], colour: str):
        """
        Reset the game and set the board to be the inputted one
        """
        self.reset_board()
        for row in range(self.rows):
            for col in range(self.cols):
                match board[row * self.cols + col]:
                    case 1:
                        self.move(row+1, col+1, "w", fast=True)
                    case -1:
                        self.move(row+1, col+1, "b", fast=True)
                    case 0:
                        pass
                    case _:
                        raise ValueError("Invalid integer in board when setting board")
        self.turn = colour
