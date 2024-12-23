"""Module containing the game class that allows dark hex to run with multiple board sizes"""

import logging
from scipy.cluster.hierarchy import DisjointSet
import util

class AbstractDarkHex:
    """
    Class that can run a game of Dark Hex. 
    
    Initialise with the dimensions and an empty board will be created. 
    
    By default, white moves first.
    
    Assume that position (1,1) is the top left cell, and (1,2) is the top row, 2nd column etc.
    
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
            "full" if the cell is occupied
        """
        # check that this player is allowed to move
        assert colour == self.turn

        cell = self.board[row][col]
        if cell != "e":  # if the chosen cell is not empty
            logging.info("Non-empty cell {cell} at {x}, {y}")
            # update our view, since we know where their piece is now
            self._get_board(colour)[row][col] = self.board[row][col]  # view update
            return "full"
        else:
            # update global board and our view
            self.board[row][col] = colour
            self._get_board(colour)[row][col] = colour
            self.turn = util.swap_colour(colour)  # swap turn
            self.update_components(row, col, colour)  # update components
            logging.info("%s played at (%s, %s)",
                         util.colour_map[colour], col, row)
            win_check = self.win_check()
            logging.info("Result of win_check is: %s", win_check)
            if win_check != "none":
                return win_check
            else:
                return "placed"
    #TODO bug where black wins but it isnt registered. shape like <

    def win_check(self) -> str:
        """
        Check if the board is in a winning position for some player.
        Returns:
            "black_win" if black has won
            "white_win" if white has won
            "none" if nobody has won
        """
        #check if the two black rows are connected or the two white columns are connected
        logging.info("Performing a win check")
        if self.black_components.connected((1,0), (self.cols, self.rows+1)):
            return "black_win"
        elif self.white_components.connected((0,0), (self.cols+1, self.rows+1)):
            return "white_win"
        else:
            return "none"

    def update_components(self, row : int, col : int, colour : str) -> None:
        """
        Update the connected components of the given colour to include the new cell (col,row)
        """
        logging.debug("Attempting to merge components surrounding (%s, %s)", col, row)
        match colour:
            case "w":
                components = self.white_components
            case "b":
                components = self.black_components
            case _:
                raise ValueError("Invalid colour given to update_components")

        components.add((col,row))
        # attempt to connect to each matching colour in surrounding hex
        adj = [
            (col-1, row), (col, row+1), (col+1, row+1), 
            (col+1, row), (col, row-1), (col-1,row-1)
        ]
        for cell in adj:
            # if adjacent cell is of the same colour
            if self.board[cell[1]][cell[0]] == colour:
                print(cell, cell[1], cell[0], self.board)
                # connect the components
                components.merge((col,row),cell)
                logging.debug("(%s, %s) and %s %s components merged", 
                              col, row, cell, util.colour_map[colour])


    def reset_board(self) -> None:
        """
        Restart the game with the same board dimensions
        """
        # reset board contents
        self.board = self._create_board()
        self.black_board = self._create_board()
        self.white_board = self._create_board()

        # revert to correct first turn
        self.turn = self.first_turn

        # set starting components
        self.black_components = DisjointSet([])
        self.white_components = DisjointSet([])
        # initial black components are top and bottom rows
        for x in range(1,self.cols+1):
            self.black_components.add((x, self.rows+1))
            self.black_components.merge((1, self.rows+1), (x, self.rows+1))
            self.black_components.add((x, 0))
            self.black_components.merge((1,0), (x,0))

        # initial white components are left and right columns
        for y in range(self.rows+2):
            self.white_components.add((0,y))
            self.white_components.merge((0,0), (0,y))
            self.white_components.add((self.cols+1, y))
            self.white_components.merge((self.cols+1,0), (self.cols+1, y))
            
        logging.info("Board reset")


    def _create_board(self) -> list[list[str]]:
        """
        Create a starting board of size cols+1 x rows+1
        The top and bottom rows are black, the left and right columns are white
        """
        logging.info("New board created")
        row = ["w"] + ["b" for i in range(self.cols)] + ["w"]
        return [row] + [
            ["w"] + ["e" for i in range(self.cols)] + ["w"] for j in range(self.rows)
        ] + [row]


    def _get_board(self, colour : str) -> list[list[str]]:
        """
        Returns the board for a given colour
        """
        match colour:
            case "b":
                return self.black_board
            case "w":
                return self.white_board
            case _:
                raise ValueError("Invalid colour given to _get_board")
