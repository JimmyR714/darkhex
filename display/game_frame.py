"""
Module for the game frame where games can be shown.
Also contains the hex buttons for within the game
"""

import tkinter as tk
import logging
import math
from functools import partial
import game.util as util


class GameFrame(tk.Frame):
    """
    Frame for showing a game in.
    """
    def __init__(self, master, frame_colour: str, num_cols: int, num_rows: int):
        super().__init__(width=(master.winfo_screenwidth() - 200) / master.num_game_frames)
        self.display_window=master
        self.frame_colour = frame_colour
        self.game_title = tk.Label(self, text="It is white's turn", pady=15)
        self.game_title.pack()
        self.turn="w"
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.hexes = {}
        self.hex_size = self.calc_hex_size(
            screen_width = self.master.winfo_screenwidth(),
            screen_height = self.master.winfo_screenheight()
        )
        #create all necessary hex buttons
        for row_index in range(num_rows+2):  #for each row
            frm_hexes = tk.Frame(self) #create row frame
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
                button = HexButton(
                    frm_hexes, self.hex_size, command=cmd, init_colour=colour
                )
                # add this button to hexes
                self.hexes[(col_index,row_index)] = button
                # place widgets
                button.grid(
                    row=row_index,
                    column=col_index,
                    padx=0,
                    pady=0,
                    sticky="nsew"
                )
            frm_hexes.place(
                x=50 + row_index*self.hex_size*math.sqrt(3)/2,
                y=50 + 2*row_index*self.hex_size
            )

    def play_move(self, row: int, col: int):
        """
        Makes a move on the board, and forces this to happen in each gameframe
        """
        #check whether it is our turn to move
        if self.frame_colour == "global" or self.frame_colour == util.colour_map[self.turn]:
            self.master.controller.update_boards(row, col, self.turn)


    def update_board(self, row: int, col: int, result: str):
        """
        Updates the display of the board to represent a move
        """
        match result:
            case "full_white":
                if self.frame_colour == "black":
                    # update the board with the new information
                    try:
                        btn = self.hexes[(col,row)]
                        btn.draw_hex(new_colour = "white")
                        logging.info("White has been discovered at (%s, %s)", col, row)
                    except KeyError:
                        logging.error("Key Error when placing cell")
            case "full_black":
                if self.frame_colour == "white":
                    # update the board with the new information
                    try:
                        btn = self.hexes[(col,row)]
                        btn.draw_hex(new_colour = "black")
                        logging.info("Black has been discovered at (%s, %s)", col, row)
                    except KeyError:
                        logging.error("Key Error when placing cell")

            case "placed":
                # update cell colour if this player learns about the update
                if self.frame_colour == "global" or self.frame_colour == util.colour_map[self.turn]:
                    try:
                        btn = self.hexes[(col,row)]
                        colour = util.colour_map[self.turn]
                        btn.draw_hex(new_colour = colour)
                        logging.info("%s has been placed at (%s, %s)", colour, col, row)
                    except KeyError:
                        logging.error("Key Error when placing cell")

                #swap turns
                self.turn = util.swap_colour(self.turn)
                self.update_title()

            case "black_win":
                logging.info("Black has won!")
                self.display_window.win_game("b")
                self.display_window.new_game()
                # finish game with black winning

            case "white_win":
                logging.info("White has won!")
                self.display_window.win_game("w")
                self.display_window.new_game()
                # finish game with white winning


    def update_title(self) -> None:
        """Updates the title with the person whose turn it is"""
        #reset win title
        self.display_window.win_title.config(text="")
        self.game_title.config(text=f"It is {util.colour_map[self.turn]}'s turn")


    def calc_hex_size(self, screen_width: float, screen_height: float) -> float:
        """
        Calculate the hex size that allows all hexes to fit.
        Sets hex size to the calculated value and returns it.
        """
        menu_width = 200 #may be wrong
        hex_width = 2*(self.num_cols+2) + (self.num_rows+2)
        num_gfs = self.master.num_game_frames
        return (1.0 / num_gfs) * 0.7 * min(
            (screen_width-menu_width)/hex_width, screen_height / (self.num_rows+2)
        )


class HexButton(tk.Canvas):
    """
    A singular hexagonal button for the game.
    Will be able to be pressed and will change colour accordingly.
    """
    BORDER_COLOUR = "gray20"

    def __init__(self, parent: tk.Frame, size: float, command, init_colour:str):
        #initialise button
        tk.Canvas.__init__(self, parent)
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
        self.create_polygon(coords, outline=self.BORDER_COLOUR, fill=colour, width=2)


    def _on_press(self, _=None):
        """
        Change canvas so it looks pressed down.
        """
        if self.command is not None:
            self.configure(relief="sunken")


    def _on_release(self, _=None):
        """
        Change canvas so it doesn't look pressed down.
        """
        if self.command is not None:
            self.configure(relief="raised")
            self.command()
