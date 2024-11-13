""" Module for the display of the program """

import tkinter as tk
import darkhex

MAX_ROWS = 11
MAX_COLS = MAX_ROWS

class VisualDarkHex:
    """
    Class that displays a game of dark hex into a tkinter frame
    
    Can have variable dimensions, up to 11x11 (for now)
    
    Can toggle between which views of the game are visible
    
    Handles all game logic through the abstract dark hex class
    """
    colour_map = {
        "w": "white",
        "b": "black",
        "g": "grey"
    }

    def __init__(self, _frame, _cols, _rows, _first_turn="w"):
        self.frame = _frame
        self.cols = _cols
        self.rows = _rows
        self.first_turn = _first_turn
        self.hexes = {}  # mapping of buttons to (col, row) cell locations
        self.game = darkhex.AbstractDarkHex(self.cols, self.rows, self.first_turn)


    def draw_view(self, colour:str):
        """
        Draws the view of the board into the frame
        
        Parameters:
            colour: Can take values "w", "b", "g" for white, black, and global
        """
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
        frm_hex = tk.Frame(self.frame)
        frm_hex.pack()
        for y, row in enumerate(board):
            for x, cell in enumerate(row):
                button = tk.Button(
                    frm_hex,
                    text="",
                    anchor=5,
                    bg=self.colour_map[cell],
                    width=5,
                    height=5
                    #command=self.play
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


def change_dim(incr: bool, label):
    """
    Change the value of the dimensions for the game
    """

    value = int(label["text"])
    label["text"] = f"{min(max(value + (2*int(incr)-1), 0), MAX_ROWS)}"

window = tk.Tk()
window.title = "Dark Hex"

# configure layout
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

# define game frame
frm_game = tk.Frame(window)

# define menu frame
frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
btn_newgame = tk.Button(frm_buttons, text="New Game")

lbl_rows = tk.Label(frm_buttons, text="Rows")
frm_rows = tk.Frame(frm_buttons)
lbl_row_value = tk.Label(master=frm_rows, text="3")
btn_row_decrease = tk.Button(master=frm_rows, text="-", command=change_dim(False, lbl_row_value))
btn_row_increase = tk.Button(master=frm_rows, text="+", command=change_dim(True, lbl_row_value))

lbl_cols = tk.Label(frm_buttons, text="Cols")
frm_cols = tk.Frame(frm_buttons)
lbl_col_value = tk.Label(master=frm_cols, text="3")
btn_col_decrease = tk.Button(master=frm_cols, text="-", command=change_dim(False, lbl_col_value))
btn_col_increase = tk.Button(master=frm_cols, text="+", command=change_dim(True, lbl_col_value))

btn_open = tk.Button(frm_buttons, text="Open")
btn_save = tk.Button(frm_buttons, text="Save As...")

# place row widgets into row frame
btn_row_decrease.grid(row=0, column=0, sticky="nsew")
lbl_row_value.grid(row=0, column=1)
btn_row_increase.grid(row=0, column=2, sticky="nsew")

# place col widgets into col frame
btn_col_decrease.grid(row=0, column=0, sticky="nsew")
lbl_col_value.grid(row=0, column=1)
btn_col_increase.grid(row=0, column=2, sticky="nsew")

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

# mainloop
window.mainloop()
