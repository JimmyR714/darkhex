"""
Module for the main display window of the program
"""

import tkinter as tk
import logging
from display.main_menu import MainMenuFrame
from display.game_frame import GameFrame
from display.training_frame import TrainingFrame
import game.util as util


class DisplayWindow(tk.Tk):
    """
    The main window for the program. 
    Upon initialization, it contains only the menu frame,
    but this can be adjusted by starting a game.
    When a game is running, it may contain multiple 
    game frames so that multiple views can be seen simultaneously.
    """

    def __init__(self, controller):
        super().__init__()
        # configure layout
        self.title = "Dark Hex"
        self.controller = controller
        self.rowconfigure(0, minsize=900, weight=1)
        self.won = False
        self.win_title = tk.Label(self, text="")
        self.win_title.grid(row=1, column=1, columnspan=3)
        self.training_mode = False

        # add key frames
        self.main_menu = MainMenuFrame(self)
        self.game_frames : list[tk.Frame] = []
        self.num_game_frames = 0
        self.training_frame = None

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
        if not self.training_mode:
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
            self.kill_game()
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


    def kill_game(self):
        """
        Resets all game frames.
        """
        #reconfigure column sizes
        for i in range(3):
            self.columnconfigure(1+i, minsize=0, weight=0)
        #remove the game frames
        for frame in self.game_frames:
            frame.grid_forget()
            frame.destroy()
        #reset our list of game frames
        self.game_frames.clear()


    def toggle_training(self):
        """
        Toggle the training mode on/off.
        Display the training menu where the game usually is when we turn it on,
        Remove the training menu when we turn it off
        """
        if self.training_mode:
            #remove frame safely
            self.columnconfigure(1, minsize=0, weight=0)
            self.training_frame.grid_forget()
            self.training_frame.destroy()
            self.training_frame = None
        else:
            #remove any games
            self.kill_game()
            #create training frame and display
            self.columnconfigure(1, minsize=self.winfo_screenwidth() - 200, weight=0)
            self.training_frame = TrainingFrame(self)
            self.training_frame.grid(row=0, column=1, sticky="nsew")
        self.training_mode = not self.training_mode


    def win_game(self, colour: str):
        """
        Outputs to the user that a game has been won, and who won it
        """
        if not self.won:
            self.won = True
            self.win_title.config(text=f"{util.colour_map[colour]} has won!")
