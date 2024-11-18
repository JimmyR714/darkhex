""" Module for the display of the program """

import tkinter as tk
import logging
from functools import partial
import darkhex
import util

MAX_ROWS = 11
MAX_COLS = 11

class DisplayWindow(tk.Tk):
    """
    Class that contains the display for the program
    
    The game can have variable dimensions, up to 11x11 (for now)
    
    Can toggle between which views of the game are visible
    
    Handles all game logic through the abstract dark hex class
    """

    def __init__(self):
        super().__init__()
        self.title = "Dark Hex"
        self.game = None
        self.hexes = {}
        self.cols = 3
        self.rows = 3

        # configure layout
        self.rowconfigure(0, minsize=800, weight=1)
        self.columnconfigure(1, minsize=800, weight=1)

        # define game frame
        self.game_frame = tk.Frame(self)
        self.hex_frame = tk.Frame(self.game_frame)
        self.game_title = tk.Label(self.game_frame, text="Create a game to start", pady=15)
        self.game_title.pack()
        self.hex_frame.pack()

        # define menu frame
        frm_buttons = tk.Frame(self, relief=tk.RAISED, bd=2)
        btn_newgame = tk.Button(frm_buttons, text="New Game", command=self.new_game)

        lbl_rows = tk.Label(frm_buttons, text="Rows")
        frm_rows = tk.Frame(frm_buttons)
        lbl_row_value = tk.Label(master=frm_rows, text=str(self.rows))
        btn_row_decrease = tk.Button(
            master=frm_rows, text="-",
            command=partial(self.change_dim, False, "rows", lbl_row_value)
        )
        btn_row_increase = tk.Button(
            master=frm_rows, text="+", command=partial(self.change_dim, True, "rows", lbl_row_value)
        )

        lbl_cols = tk.Label(frm_buttons, text="Cols")
        frm_cols = tk.Frame(frm_buttons)
        lbl_col_value = tk.Label(master=frm_cols, text=str(self.cols))
        btn_col_decrease = tk.Button(
            master=frm_cols, text="-",
            command=partial(self.change_dim, False, "cols", lbl_col_value)
        )
        btn_col_increase = tk.Button(
            master=frm_cols, text="+", command=partial(self.change_dim, True, "cols", lbl_col_value)
        )

        btn_open = tk.Button(frm_buttons, text="Open")
        btn_save = tk.Button(frm_buttons, text="Save As...")

        # place row widgets into row frame
        btn_row_decrease.grid(row=0, column=0, sticky="ew")
        lbl_row_value.grid(row=0, column=1, sticky="ew")
        btn_row_increase.grid(row=0, column=2, sticky="ew")

        # place col widgets into col frame
        btn_col_decrease.grid(row=0, column=0, sticky="ew")
        lbl_col_value.grid(row=0, column=1, sticky="ew")
        btn_col_increase.grid(row=0, column=2, sticky="ew")

        # place widgets into menu frame
        btn_newgame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        lbl_rows.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        frm_rows.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        lbl_cols.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        frm_cols.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        btn_open.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        btn_save.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        # place two main frames
        frm_buttons.grid(row=0, column=0, sticky="ns")
        self.game_frame.grid(row=0, column=1, sticky="nsew")


    def change_dim(self, incr: bool, dim : str, label) -> None:
        """
        Change the value of the dimensions for the game
        """
        match dim:
            case "rows":
                self.rows = min(max(self.rows + (2*int(incr)-1), 1), MAX_ROWS)
                x = self.rows
            case "cols":
                self.cols = min(max(self.cols + (2*int(incr)-1), 1), MAX_ROWS)
                x = self.cols

        label["text"] = f"{x}"


    def new_game(self) -> None:
        """
        Start a new game of dark hex
        """
        #remove the previous game
        widgets = self.hex_frame.grid_slaves()
        for l in widgets:
            l.destroy()

        self.hexes = {}
        self.game = darkhex.AbstractDarkHex(self.cols, self.rows)
        self.update_title()
        self.draw_view("g")


    def draw_view(self, colour:str) -> None:
        """
        Draws the view of the board into the frame
        
        Parameters:
            colour: Can take values "w", "b", "g" for white, black, and global
        """
        # check that a game is running
        assert self.game is not None

        # choose the correct board
        match colour:
            case "w":
                board = self.game.white_board
                logging.info("White board drawn")
            case "b":
                board = self.game.black_board
                logging.info("Black board drawn")
            case "g":
                board = self.game.board
                logging.info("Global board drawn")
            case _:
                raise ValueError("Invalid colour input to draw_view")

        # now draw the chosen board
        cnv_hex = tk.Canvas(self.hex_frame)
        for row_index in range(self.rows+2):
            row = board[row_index]
            for col_index in range(self.cols+2):
                cell = row[col_index]
                row_num = row_index
                col_num = row_index + 2*(col_index-1)
                hex_x, hex_y = col_num*55, row_num*65
                #check if this is a bordering white cell
                if col_index in [0, self.cols+1]:
                    util.draw_hex((hex_x+26,hex_y+35), 45, cnv_hex, bkg_colour="white")
                    button = tk.Button(
                        self.hex_frame,
                        bg="white",
                        width=5,
                        height=3,
                    )
                    button.grid(
                        row=row_num,
                        column=col_num,
                        padx=5,
                        pady=5,
                        sticky="nsew"
                    )
                #check if this is a bordering black cell
                elif row_index in [0, self.rows+1]:
                    util.draw_hex((hex_x+26,hex_y+35), 45, cnv_hex, bkg_colour="black")
                    button = tk.Button(
                        self.hex_frame,
                        bg="black",
                        width=5,
                        height=3,
                    )
                    button.grid(
                        row=row_num,
                        column=col_num,
                        padx=5,
                        pady=5,
                        sticky="nsew"
                    )
                else:
                    # create the button
                    button = tk.Button(
                        self.hex_frame,
                        bg=util.colour_map[cell],
                        width=5,
                        height=3,
                        command=partial(self.play_move, row_index, col_index)
                    )
                    # add this button to hexes
                    self.hexes[(col_index,row_index)] = button
                    # create the hex
                    # place widgets
                    button.grid(
                        row=row_num,
                        column=col_num,
                        padx=5,
                        pady=5,
                        sticky="nsew"
                    )
                    logging.debug("Button gridded to (%s, %s)", col_num, row_num)
                    util.draw_hex((hex_x+26,hex_y+35), 45, cnv_hex)
        cnv_hex.place(x=0,y=0)


    def play_move(self, row : int, col : int) -> None:
        """
        Play a move on the board
        """
        turn = self.game.turn
        result = self.game.move(row, col, turn)
        match result:
            case "full":
                # allow them to retry
                pass
            case "placed":
                # update cell colour, swap turns
                try:
                    btn = self.hexes[(col,row)]
                    colour = util.swap_colour(self.game.turn)
                    logging.info("%s has been placed at (%s, %s)", colour, col, row)
                    btn.config(bg=util.colour_map[colour])
                    self.update_title()
                except KeyError:
                    logging.error("Key Error when placing cell")
            case "black_win":
                logging.info("Black has won!")
                self.new_game()
                # finish game with black winning
            case "white_win":
                logging.info("White has won!")
                self.new_game()
                # finish game with white winning


    def update_title(self) -> None:
        """Updates the title with the person whose turn it is"""
        self.game_title.config(text=f"It is {util.colour_map[self.game.turn]}'s turn")


def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    window = DisplayWindow()
    window.mainloop()

if __name__ == "__main__":
    main()
