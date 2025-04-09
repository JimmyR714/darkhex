"""
Module for everything displayed on the screen
"""

import logging
import math
from functools import partial
import tkinter as tk
from main import Controller
import game.util as util

MAX_ROWS = 11
MAX_COLS = 11
MAX_DEPTH = 10
MAX_BELIEFS = 1024
MAX_ITERS = 25

class DisplayWindow(tk.Tk):
    """
    The main window for the program. 
    Upon initialization, it contains only the menu frame,
    but this can be adjusted by starting a game.
    When a game is running, it may contain multiple 
    game frames so that multiple views can be seen simultaneously.
    """

    def __init__(self, controller: Controller):
        super().__init__()
        # configure layout
        self.title = "Dark Hex"
        self.controller = controller
        self.rowconfigure(0, minsize=900, weight=1)
        self.won = False
        self.win_title = tk.Label(self, text="")
        self.win_title.grid(row=1, column=1, columnspan=3)

        # add key frames
        self.main_menu = MainMenuFrame(self)
        self.game_frames : list[tk.Frame] = []
        self.num_game_frames = 0

        # grid frames
        self.fullscreen = True
        self.attributes("-fullscreen", self.fullscreen)
        self.main_menu.grid(row=0, column=0, sticky="ns")

        #window settings
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.end_fullscreen)


    def toggle_fullscreen(self, _=None):
        """
        Toggle between a fullscreen and non-fullscreen window
        """
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        return "break"


    def end_fullscreen(self, _=None):
        """
        Turn fullscreen off
        """
        self.fullscreen = False
        self.attributes("-fullscreen", False)
        return "break"


    def new_game(self):
        """
        First gets the selected game settings from the main menu.
        Then it creates a new game, with the correct sizes and game frames.
        """
        # get rows and cols
        rows = self.main_menu.rows
        cols = self.main_menu.cols

        # get the required displays
        displays = []
        if self.main_menu.global_display.get() == 1:
            displays.append("global")
        if self.main_menu.white_display.get() == 1:
            displays.append("white")
        if self.main_menu.black_display.get() == 1:
            displays.append("black")

        #reset previous game frames
        for i in range(3):
            self.columnconfigure(1+i, minsize=0, weight=0)
        for frame in self.game_frames:
            frame.grid_forget()
            frame.destroy()
        self.game_frames.clear()
        #set the number of game frames to be created for correct sizing
        self.num_game_frames = len(displays)
        #create new game frames
        for d in displays:
            self.game_frames.append(
                GameFrame(master=self, frame_colour=d, num_cols=cols, num_rows=rows)
            )
        #place each game frame
        gf_size = (self.winfo_screenwidth() - 200) / self.num_game_frames
        for i, gf in enumerate(self.game_frames):
            self.columnconfigure(1+i, minsize=gf_size, weight=0)
            gf.grid(row=0, column=1+i, sticky="nsew")

        game_type = self.main_menu.game_selection.get()
        #make the game
        self.controller.new_game(num_cols=cols, num_rows=rows)
        match game_type:
            case "Player vs Player":
                logging.debug("Player vs Player game created")
            case "Player vs Agent":
                agent_settings = self.main_menu.agent_settings_1
                agent_settings["type"] = self.main_menu.agent_selection_1.get()
                agent_settings["colour"] = self.main_menu.agent_colour.get()
                self.controller.new_pva_game(agent_settings)
                logging.debug("Player vs Agent game created")
            case "Agent vs Agent":
                agent_1_settings = self.main_menu.agent_settings_1
                agent_1_settings["colour"] = self.main_menu.agent_colour.get()
                agent_1_settings["type"] = self.main_menu.agent_selection_1.get()
                agent_2_settings = self.main_menu.agent_settings_2
                agent_2_settings["colour"] = util.swap_colour(agent_1_settings["colour"])
                agent_2_settings["type"] = self.main_menu.agent_selection_2.get()
                iterations = self.main_menu.iterations
                #run the new game in the controller
                logging.debug("Agent vs Agent game created")
                self.controller.new_ava_game(agent_1_settings, agent_2_settings, iterations)


    def win_game(self, colour: str):
        """
        Outputs to the user that a game has been won, and who won it
        """
        if not self.won:
            self.won = True
            self.win_title.config(text=f"{util.colour_map[colour]} has won!")

class MainMenuFrame(tk.Frame):
    """
    Frame that contains the main menu for the program.
    Remains on the left of the screen throughout gameplay
    """
    #general variables
    rows = 3
    cols = 3
    depth = [3,3]
    beliefs = [16,16]
    iterations = 7

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
            command=partial(self.change_lbl, False, "rows", lbl_row_value)
        )
        btn_row_increase = tk.Button(
            master=frm_rows, text="+", command=partial(self.change_lbl, True, "rows", lbl_row_value)
        )

        #column changing
        lbl_cols = tk.Label(self, text="Cols")
        frm_cols = tk.Frame(self)
        lbl_col_value = tk.Label(master=frm_cols, text=str(self.cols))
        btn_col_decrease = tk.Button(
            master=frm_cols, text="-",
            command=partial(self.change_lbl, False, "cols", lbl_col_value)
        )
        btn_col_increase = tk.Button(
            master=frm_cols, text="+", command=partial(self.change_lbl, True, "cols", lbl_col_value)
        )

        # place row widgets into row frame
        btn_row_decrease.grid(row=0, column=0, sticky="ew")
        lbl_row_value.grid(row=0, column=1, sticky="ew")
        btn_row_increase.grid(row=0, column=2, sticky="ew")

        # place col widgets into col frame
        btn_col_decrease.grid(row=0, column=0, sticky="ew")
        lbl_col_value.grid(row=0, column=1, sticky="ew")
        btn_col_increase.grid(row=0, column=2, sticky="ew")

        #display selection
        lbl_displays = tk.Label(self, text="Displays")
        frm_displays = tk.Frame(self)
        self.global_display = tk.IntVar(value=1)
        btn_global_display = tk.Checkbutton(
            frm_displays,
            text="Global",
            variable=self.global_display,
            onvalue=1,
            offvalue=0,
            height=2,
            width=10
        )
        self.white_display = tk.IntVar()
        btn_white_display = tk.Checkbutton(
            frm_displays,
            text="White",
            variable=self.white_display,
            onvalue=1,
            offvalue=0,
            height=2,
            width=10
        )
        self.black_display = tk.IntVar()
        btn_black_display = tk.Checkbutton(
            frm_displays,
            text="Black",
            variable=self.black_display,
            onvalue=1,
            offvalue=0,
            height=2,
            width=10
        )

        # place display selection widgets into frame
        btn_global_display.pack()
        btn_white_display.pack()
        btn_black_display.pack()

        #game type selection
        game_options = [
            "Player vs Player",
            "Player vs Agent",
            "Agent vs Agent"
        ]
        lbl_game_type = tk.Label(self, text="Select Game:")
        frm_game_type = tk.Frame(self)
        self.game_selection = tk.StringVar()
        #change below to switch default game type
        self.game_selection.set("Player vs Player")
        ddm_game = tk.OptionMenu(
            frm_game_type,
            self.game_selection,
            *game_options,
            command=partial(self.update_game_menu)
        )

        #place game selection widgets into frame
        ddm_game.pack()

        #variables used in some game types
        self.agent_selection_1 = tk.StringVar(value="General")
        self.agent_selection_2 = tk.StringVar(value="General")
        self.agent_colour = tk.StringVar(value="w")
        self.agent_settings_1 = {} #agent specific settings
        self.agent_settings_2 = {}

        #menu that appears based on game type
        self.frm_game_settings = tk.Frame(self)

        #menu that appears based on agent type
        self.frm_agent_settings_1 : tk.Frame = None
        self.frm_agent_settings_2 : tk.Frame = None

        #set default agent settings
        self.update_agent_menu(1, "General")
        self.update_agent_menu(2, "General")


        # place widgets into menu frame
        btn_newgame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        lbl_rows.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        frm_rows.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        lbl_cols.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        frm_cols.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        lbl_displays.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        frm_displays.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        lbl_game_type.grid(row=7, column=0, sticky="ew", padx=5, pady=5)
        frm_game_type.grid(row=8, column=0, sticky="ew", padx=5, pady=5)
        self.frm_game_settings.grid(row=9, column=0, sticky="ew", padx=5, pady=5)


    def change_lbl(self, incr: bool, dim : str, label: tk.Label) -> None:
        """
        Change the value of a label
        """
        match dim:
            case "rows":
                self.rows = min(max(self.rows + (2*int(incr)-1), 1), MAX_ROWS)
                x = self.rows
            case "cols":
                x = self.cols = min(max(self.cols + (2*int(incr)-1), 1), MAX_COLS)
            case "depth_1":
                x = self.depth[0] = min(max(self.depth[0] + (2*int(incr)-1), 1), MAX_DEPTH)
            case "depth_2":
                x = self.depth[1] = min(max(self.depth[1] + (2*int(incr)-1), 1), MAX_DEPTH)
            case "beliefs_1":
                #this one doubles and halves each click
                x = self.beliefs[0] = int(
                    min(max(self.beliefs[0] * (3*int(incr)+1)/2, 1), MAX_BELIEFS)
                )
            case "beliefs_2":
                x = self.beliefs[1] = int(
                    min(max(self.beliefs[1] * (3*int(incr)+1)/2, 1), MAX_BELIEFS)
                )
            case "iters":
                x = self.iterations = int(
                    min(max(self.iterations + 2*(2*int(incr)-1), 1), MAX_ITERS)
                )

        #update the correct label with the new text
        label["text"] = f"{x}"


    def update_game_menu(self, game_type: str):
        """
        Change the menus that display once we change our game type selection
        """
        for widget in self.frm_game_settings.winfo_children():
            widget.destroy()
        # agent selection
        # ADD NEW AGENT HERE
        agent_options = [
            "General",
            "Basic",
            "RL",
            "Abstract"
        ]
        match game_type:
            case "Player vs Player":
                #place widgets required for a player vs player game
                pass
            case "Player vs Agent":
                #place widgets required for a single agent selection
                tk.Label(self.frm_game_settings, text="Select Agent:").pack()
                tk.OptionMenu(
                    self.frm_game_settings,
                    self.agent_selection_1,
                    *agent_options,
                    command=partial(self.update_agent_and_game_menu, 1)
                ).pack()
                self.frm_agent_settings_1 = tk.Frame(self.frm_game_settings)
                self.update_agent_menu(1, self.agent_selection_1.get())
                self.frm_agent_settings_1.pack()
                tk.Label(self.frm_game_settings, text="Agent Colour:").pack()
                tk.Radiobutton(
                    self.frm_game_settings,
                    variable=self.agent_colour,
                    text = "White",
                    value="w"
                ).pack()
                tk.Radiobutton(
                    self.frm_game_settings,
                    variable=self.agent_colour,
                    text = "Black",
                    value="b"
                ).pack()
            case "Agent vs Agent":
                #place widgets required for two agent selections
                tk.Label(self.frm_game_settings, text="Iterations").pack()
                lbl_iters_value = tk.Label(self.frm_game_settings, text=str(self.iterations))
                lbl_iters_value.pack()
                tk.Button(
                    master=self.frm_game_settings, text="-",
                    command=partial(self.change_lbl, False, "iters", lbl_iters_value)
                ).pack()
                tk.Button(
                    master=self.frm_game_settings, text="+",
                    command=partial(self.change_lbl, True, "iters", lbl_iters_value)
                ).pack()
                tk.Label(self.frm_game_settings, text="Select Agent 1:").pack()
                tk.OptionMenu(
                    self.frm_game_settings,
                    self.agent_selection_1,
                    *agent_options,
                    command=partial(self.update_agent_and_game_menu, 1)
                ).pack()
                self.frm_agent_settings_1 = tk.Frame(self.frm_game_settings)
                self.update_agent_menu(1, self.agent_selection_1.get())
                self.frm_agent_settings_1.pack()
                tk.Label(self.frm_game_settings, text="Select Agent 2:").pack()
                tk.OptionMenu(
                    self.frm_game_settings,
                    self.agent_selection_2,
                    *agent_options,
                    command=partial(self.update_agent_and_game_menu, 2)
                ).pack()
                self.frm_agent_settings_2 = tk.Frame(self.frm_game_settings)
                self.update_agent_menu(2, self.agent_selection_2.get())
                self.frm_agent_settings_2.pack()
                tk.Label(self.frm_game_settings, text="Agent 1 Colour:").pack()
                tk.Radiobutton(
                    self.frm_game_settings,
                    variable=self.agent_colour,
                    text = "White",
                    value="w"
                ).pack()
                tk.Radiobutton(
                    self.frm_game_settings,
                    variable=self.agent_colour,
                    text = "Black",
                    value="b"
                ).pack()


    def update_agent_and_game_menu(self, agent_num: int, agent_type: str):
        """
        Updates the game menu and agent menu, aligning them correctly
        """
        match agent_num:
            case 1:
                self.agent_selection_1.set(agent_type)
            case 2:
                self.agent_selection_2.set(agent_type)
            case _:
                raise ValueError("Invalid agent number")
        self.update_game_menu(self.game_selection.get())


    def update_agent_menu(self, agent_num : int, agent_type: str = "General"):
        """
        Change the agent menu that displays once we change our agent selection
        """
        #choose the correct menu and settings to update
        if agent_num == 1:
            settings = self.agent_settings_1
            frm = self.frm_agent_settings_1
        else:
            settings = self.agent_settings_2
            frm = self.frm_agent_settings_2
        if frm is not None:
            #reset the previous game settings
            for widget in frm.winfo_children():
                widget.destroy()
        #update menu based on selection
        # ADD AGENT SETTINGS HERE
        match agent_type:
            case "General":
                #parameter currently does nothing
                settings["difficulty"] = tk.IntVar(value=3)
            case "Basic":
                settings["depth"] = tk.IntVar(value=3)
                settings["beliefs"] = tk.IntVar(value=20)
                #depth changing
                tk.Label(frm, text="Depth").grid(row=0, column=1, sticky="ew")
                lbl_depth_value = tk.Label(master=frm, text=str(self.depth[agent_num-1]))
                tk.Button(
                    master=frm, text="-",
                    command=partial(self.change_lbl, False, f"depth_{agent_num}", lbl_depth_value)
                ).grid(row=1, column=0, sticky="ew")
                lbl_depth_value.grid(row=1, column=1, sticky="ew")
                tk.Button(
                    master=frm, text="+",
                    command=partial(self.change_lbl, True, f"depth_{agent_num}", lbl_depth_value)
                ).grid(row=1, column=2, sticky="ew")
                #beliefs changing
                tk.Label(frm, text="Beliefs").grid(row=2, column=1, sticky="ew")
                lbl_beliefs_value = tk.Label(master=frm, text=str(self.beliefs[agent_num-1]))
                tk.Button(
                    master=frm, text="-",
                    command=partial(
                        self.change_lbl, False, f"beliefs_{agent_num}", lbl_beliefs_value
                    )
                ).grid(row=3, column=0, sticky="ew")
                lbl_beliefs_value.grid(row=3, column=1, sticky="ew")
                tk.Button(
                    master=frm, text="+",
                    command=partial(
                        self.change_lbl, True, f"beliefs_{agent_num}", lbl_beliefs_value
                    )
                ).grid(row=3, column=2, sticky="ew")
            case "Abstract":
                #TODO create agent settings menu
                settings["width_1"] = tk.IntVar()


class GameFrame(tk.Frame):
    """
    Frame for showing a game in.
    """
    def __init__(self, master: DisplayWindow, frame_colour: str, num_cols: int, num_rows: int):
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
