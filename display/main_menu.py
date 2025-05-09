"""
Module for the leftmost menu in the program
"""

import tkinter as tk
from functools import partial


MAX_ROWS = 11
MAX_COLS = 11
MAX_DEPTH = 10
MAX_BELIEFS = 1024
MAX_ITERS = 25
MAX_WIDTH = 10
MAX_BOARDS = 100


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
    width_1 = [5,5]
    width_2_vc = [3,3]
    width_2_semi_vc = [2,2]
    fake_boards = [10,10]
    iterations = 7

    def __init__(self, master):
        super().__init__(master=master, relief=tk.RAISED, bd=2)
        # define widgets for main menu frame
        # new game button
        btn_newgame = tk.Button(self, text="New Game", command=master.new_game)
        btn_training = tk.Button(self, text="Training", command=master.toggle_training)

        # row changing
        frm_rows = tk.Frame(self)
        self.add_parameter(
            frm=frm_rows,
            lbl_txt="Rows",
            lbl_init_val=str(self.rows),
            update="rows"
        )

        #column changing
        frm_cols = tk.Frame(self)
        self.add_parameter(
            frm=frm_cols,
            lbl_txt="Cols",
            lbl_init_val=str(self.cols),
            update="cols"
        )

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
        btn_training.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        frm_rows.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
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
                x = self.rows = min(max(self.rows + (2*int(incr)-1), 1), MAX_ROWS)
            case "cols":
                x = self.cols = min(max(self.cols + (2*int(incr)-1), 1), MAX_COLS)
            case "iters":
                x = self.iterations = int(
                    min(max(self.iterations + 2*(2*int(incr)-1), 1), MAX_ITERS)
                )
            case _:
                n = int(dim[-1]) - 1
                if "depth" in dim:
                    x = self.depth[n] = min(max(self.depth[n] + (2*int(incr)-1), 1), MAX_DEPTH)
                elif "beliefs" in dim:
                    x = self.beliefs[n] = int(
                        min(max(self.beliefs[n] * (3*int(incr)+1)/2, 1), MAX_BELIEFS)
                    )
                elif "width_1" in dim:
                    x = self.width_1[n] = min(max(self.width_1[n] + (2*int(incr)-1), 1), MAX_WIDTH)
                elif "width_2_vc" in dim:
                    x = self.width_2_vc[n] = min(
                        max(self.width_2_vc[n] + (2*int(incr)-1), 1), MAX_WIDTH
                    )
                elif "width_2_semi_v" in dim:
                    x = self.width_2_semi_vc[n] = min(
                        max(self.width_2_semi_vc[n] + (2*int(incr)-1), 1), MAX_WIDTH
                    )
                elif "fake_boards" in dim:
                    x = self.fake_boards[n] = min(
                        max(self.fake_boards[n] + 4*(2*int(incr)-1), 2), MAX_BOARDS
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
            "Bayesian"
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


    def add_parameter(self, frm: tk.Frame, lbl_txt: str, lbl_init_val: str, update: str):
        """
        Add some agent parameter adjustable by up/down buttons with a max/min.
        """
        tk.Label(frm, text=lbl_txt).grid(row=0, column=1, sticky="ew")
        lbl_value = tk.Label(master=frm, text=lbl_init_val)
        tk.Button(
            master=frm, text="-",
            command=partial(self.change_lbl, False, update, lbl_value)
        ).grid(row=1, column=0, sticky="ew")
        lbl_value.grid(row=1, column=1, sticky="ew")
        tk.Button(
            master=frm, text="+",
            command=partial(self.change_lbl, True, update, lbl_value)
        ).grid(row=1, column=2, sticky="ew")


    def add_checkbox(self, frm: tk.Frame, lbl_txt: str, tk_var: tk.Variable):
        """
        Add checkbox parameter.
        """
        tk.Label(frm, text=lbl_txt).grid(row=0, column=0, sticky="ew")
        tk.Checkbutton(frm, variable=tk_var).grid(row=0, column=1, sticky="ew")


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
                frm_depth = tk.Frame(frm)
                self.add_parameter(
                    frm=frm_depth,
                    lbl_txt="Depth",
                    lbl_init_val=str(self.depth[agent_num-1]),
                    update=f"depth_{agent_num}"
                )
                frm_depth.grid(row=1, sticky="ew")
                #beliefs changing
                frm_beliefs = tk.Frame(frm)
                self.add_parameter(
                    frm=frm_beliefs,
                    lbl_txt="Beliefs",
                    lbl_init_val=str(self.beliefs[agent_num-1]),
                    update=f"beliefs_{agent_num}"
                )
                frm_beliefs.grid(row=2, sticky="ew")
            case "Bayesian":
                #add up/down parameters
                settings["width_1"] = tk.IntVar(value=5)
                settings["width_2_vc"] = tk.IntVar(value=3)
                settings["width_2_semi_vc"] = tk.IntVar(value=2)
                settings["fake_boards"] = tk.IntVar(value=10)
                frm_width_1 = tk.Frame(frm)
                frm_width_2_vc = tk.Frame(frm)
                frm_width_2_semi_vc = tk.Frame(frm)
                frm_fake_boards = tk.Frame(frm)
                self.add_parameter(frm_width_1, "Width 1",
                        str(self.width_1[agent_num-1]), f"width_1_{agent_num}")
                self.add_parameter(frm_width_2_vc, "Width 2 VC",
                        str(self.width_2_vc[agent_num-1]), f"width_2_vc_{agent_num}")
                self.add_parameter(frm_width_2_semi_vc, "Width 2 SVC",
                        str(self.width_2_semi_vc[agent_num-1]), f"width_2_semi_vc_{agent_num}")
                self.add_parameter(frm_fake_boards, "Fake Boards",
                        str(self.fake_boards[agent_num-1],), f"fake_boards_{agent_num}")
                frm_width_1.grid(row=1, sticky="ew")
                frm_width_2_vc.grid(row=2, sticky="ew")
                frm_width_2_semi_vc.grid(row=3, sticky="ew")
                frm_fake_boards.grid(row=4, sticky="ew")
                #add learning parameter
                settings["learning"] = tk.BooleanVar(value=False)
                frm_learning = tk.Frame(frm)
                self.add_checkbox(
                    frm=frm_learning,
                    lbl_txt="Online",
                    tk_var=settings["learning"]
                )
                frm_learning.grid(row=5, sticky="ew")
