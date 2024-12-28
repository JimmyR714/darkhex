import logging
import math
from functools import partial
import tkinter as tk

MAX_ROWS = 11
MAX_COLS = 11

class DisplayWindow(tk.Tk):
    """
    The main window for the program. 
    Upon initialisation, it contains only the menu frame,
    but this can be adjusted by starting a game.
    When a game is running, it may contain multiple 
    game frames so that multiple views can be seen simultaneously.
    """

    def __init__(self):
        super().__init__()
        # configure layout
        self.rowconfigure(0, minsize=800, weight=1)
        self.columnconfigure(1, minsize=800, weight=1)

        # add key frames
        self.main_menu = MainMenuFrame(self)
        self.game_frame = tk.Frame(self)

        # grid frames
        self.main_menu.grid(row=0, column=0, sticky="ns")
        self.game_frame.grid(row=0, column=1, sticky="nsew")


    def new_game(self):
        """
        Create a new game of dark hex
        """
        # get the selected game settings
        rows = self.main_menu.rows
        cols = self.main_menu.cols

        #TODO allow multiple gameframes to be displayed
        self.game_frame = GameFrame(num_cols=cols, num_rows=rows)
        self.game_frame.grid(row=0, column=1, sticky="nsew")


class MainMenuFrame(tk.Frame):
    rows = 3
    cols = 3
    def __init__(self, master: DisplayWindow):
        super().__init__(master=master, relief=tk.RAISED, bd=2)
        # define widgets for main menu frame
        # new game button
        btn_newgame = tk.Button(self, text="New Game", command=master.new_game)

        # row changing
        lbl_rows = tk.Label(self, text="Rows")
        frm_rows = tk.Frame(self)
        lbl_row_value = tk.Label(master=frm_rows, text=str(self.rows))
        btn_row_decrease = tk.Button(
            master=frm_rows, text="-",
            command=partial(self.change_dim, False, "rows", lbl_row_value)
        )
        btn_row_increase = tk.Button(
            master=frm_rows, text="+", command=partial(self.change_dim, True, "rows", lbl_row_value)
        )

        #column changing
        lbl_cols = tk.Label(self, text="Cols")
        frm_cols = tk.Frame(self)
        lbl_col_value = tk.Label(master=frm_cols, text=str(self.cols))
        btn_col_decrease = tk.Button(
            master=frm_cols, text="-",
            command=partial(self.change_dim, False, "cols", lbl_col_value)
        )
        btn_col_increase = tk.Button(
            master=frm_cols, text="+", command=partial(self.change_dim, True, "cols", lbl_col_value)
        )

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


    def change_dim(self, incr: bool, dim : str, label: tk.Label) -> None:
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


class GameFrame(tk.Frame):
    """
    Frame for showing a game in.
    """
    def __init__(self, num_cols: int, num_rows: int):
        super().__init__()
        self.hexes = {}

        #TODO calculate hex size and centres
        SIZE = 40

        #create all necessary hex buttons
        for row_index in range(num_rows+2):  #for each row
            for col_index in range(num_cols+2):
                #check if this is a bordering white cell
                if col_index == 0 or col_index == num_cols+1:
                    cmd = None
                    colour = "white"
                #check if this is a bordering black cell
                elif row_index == 0 or row_index == num_rows+1:
                    cmd = None
                    colour = "black"
                else:
                    cmd = partial(self.play_move, row_index, col_index)
                    colour = "grey"
                # create the hex
                button = HexButton(self, SIZE, command=cmd, init_colour=colour)
                # add this button to hexes
                self.hexes[(col_index,row_index)] = button
                # place widgets
                button.grid(
                    row=row_index,
                    column=2*col_index + row_index,
                    padx=0,
                    pady=0,
                    sticky="nsew"
                )


    def play_move(self, row: int, col: int):
        pass


class HexButton(tk.Canvas):
    """
    A singular hexagonal button for the game.
    Will be able to be pressed and will change colour accordingly.
    """
    BORDER_COLOUR = "black"

    def __init__(self, parent: tk.Frame, size: int, command, init_colour:str):
        #initialise button
        tk.Canvas.__init__(self, parent, borderwidth=1, relief="raised", highlightthickness=0)
        self.command = command
        self.colour = init_colour
        self.size = size
        self.configure(width = math.sqrt(3) * size, height= 2 * size)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.draw_hex()


    def draw_hex(self, new_colour: str=None) -> int:
        """Function to draw a hexagon"""
        def vertex(i):
            """Returns one vertex of the hexagon"""
            angle_deg = (60 * i) - 30
            angle_rad = (math.pi / 180) * angle_deg
            #we return the expression below because the centre of the canvas
            #is at self.size, and the vertex comes from the trig expression
            return (self.size * (1 + math.cos(angle_rad)),
                self.size * (1 + math.sin(angle_rad))
            )

        # collect coords for the hexagon
        coords = []
        for i in range(7):
            vertex_i = vertex(i%6)
            coords.append(vertex_i[0])
            coords.append(vertex_i[1])
        if new_colour is None:
            colour = self.colour
        else:
            colour = new_colour
        self.create_polygon(coords, outline=self.BORDER_COLOUR, fill=colour, width=1)
        logging.debug("Hex drawn with coords %s", coords)

    #TODO change so it is visible
    def _on_press(self, event):
        """
        Change canvas so it looks pressed down.
        """
        self.configure(relief="sunken")

    #TODO change so it is visible
    def _on_release(self, event):
        """
        Change canvas so it doesn't look pressed down.
        """
        self.configure(relief="raised")
        if self.command is not None:
            self.command()

def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    window = DisplayWindow()
    window.mainloop()

if __name__ == "__main__":
    main()
