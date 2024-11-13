""" Module for the display of the program """

import tkinter as tk
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
        frm_game = tk.Frame(self)
        self.game_frame = frm_game

        # define menu frame
        frm_buttons = tk.Frame(self, relief=tk.RAISED, bd=2)
        btn_newgame = tk.Button(frm_buttons, text="New Game", command=self.new_game())

        lbl_rows = tk.Label(frm_buttons, text="Rows")
        frm_rows = tk.Frame(frm_buttons)
        lbl_row_value = tk.Label(master=frm_rows, text=str(self.rows))
        btn_row_decrease = tk.Button(
            master=frm_rows, text="-", command=self.change_dim(False, "rows", lbl_row_value)
        )
        btn_row_increase = tk.Button(
            master=frm_rows, text="+", command=self.change_dim(True, "rows", lbl_row_value)
        )

        lbl_cols = tk.Label(frm_buttons, text="Cols")
        frm_cols = tk.Frame(frm_buttons)
        lbl_col_value = tk.Label(master=frm_cols, text=str(self.cols))
        btn_col_decrease = tk.Button(
            master=frm_cols, text="-", command=self.change_dim(False, "cols", lbl_col_value)
        )
        btn_col_increase = tk.Button(
            master=frm_cols, text="+", command=self.change_dim(True, "cols", lbl_col_value)
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
        frm_game.grid(row=0, column=1, sticky="nsew")


    def change_dim(self, incr: bool, dim : str, label):
        """
        Change the value of the dimensions for the game
        """
        match dim:
            case "rows":
                self.rows = min(max(self.rows + (2*int(incr)-1), 0), MAX_ROWS)
                x = self.rows
            case "cols":
                self.cols = min(max(self.cols + (2*int(incr)-1), 0), MAX_ROWS)
                x = self.cols

        label["text"] = f"{x}"


    def new_game(self):
        """
        Start a new game of dark hex
        """
        self.hexes = {}
        self.game = darkhex.AbstractDarkHex(self.cols, self.rows)
        self.draw_view("g")


    def draw_view(self, colour:str):
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
            case "b":
                board = self.game.black_board
            case "g":
                board = self.game.board
            case _:
                raise ValueError("Invalid colour input to draw_view")

        # now draw the chosen board
        frm_hex = tk.Frame(self.game_frame)
        frm_hex.pack()
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                if x == 0 or x == self.rows - 1 or y == 0 or y == self.cols - 1:
                    continue
                button = tk.Button(
                    frm_hex,
                    text=f"{x}, {y}",
                    anchor="center",
                    bg=util.colour_map[cell],
                    width=5,
                    height=5,
                    command=self.play_move(x,y)
                )
                # add this button to hexes
                self.hexes[button] = (x,y)
                button.grid(
                    row=y,
                    column=x,
                    padx=5,
                    pady=5,
                    sticky="nsew"
                )


    def play_move(self, x, y):
        """
        Play a move on the board
        """
        turn = self.game.turn
        result = self.game.move(x, y, turn)
        match result:
            case "full":
                # allow them to retry
                pass
            case "placed":
                # update cell colour, swap turns
                try:
                    btn = self.hexes[(x,y)]
                    colour = self.game.board[y][x]
                    btn.bg=util.colour_map[util.swap_colour(colour)]
                except KeyError:
                    pass
            case "black_win":
                # finish game with black winning
                pass
            case "white_win":
                # finish game with white winning
                pass


def main():
    """
    Create the display for the game and run the mainloop
    """
    window = DisplayWindow()
    window.mainloop()

if __name__ == "__main__":
    main()
