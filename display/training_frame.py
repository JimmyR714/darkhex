"""
Module for the training frame where we can train and simulate agents
"""

import tkinter as tk
from functools import partial


class TrainingFrame(tk.Frame):
    """
    Frame where we can initiate simulations between agents, and display the results.
    """
    def __init__(self, master):
        super().__init__(master=master, relief=tk.RAISED, bd=2)
        #add title
        tk.Label(master=self, text="Training Frame").grid(row=0, column=0, sticky="nw")
        self.iters = tk.StringVar()
        tk.Label(master=self, text="Iterations").grid(row=1, column=0, sticky="nw")
        tk.Entry(self, textvariable=self.iters).grid(row=2, column=0, sticky="nw")
        sim_options = [
            "Offline",
            "Online vs Offline",
            "Online"
        ]
        self.sim_type = tk.StringVar(value="Offline")
        tk.OptionMenu(
            self,
            self.sim_type,
            *sim_options,
            #command=partial(self.update_agent_and_game_menu, 1)
        ).grid(row=3, column=0, sticky="nw")
        #allow us to add an agent to list of agents to be run
        self.frm_add_agent = tk.Frame(master=self)
        agent_types = [
            "General",
            "Belief",
            "RL",
            "Bayesian",
            "From File"
        ]
        self.agent_type = tk.StringVar(value="General")
        self.agent_settings = {}
        self.frm_agent_settings = None
        tk.Label(master=self.frm_add_agent, text="Select Agent Type:").grid(row=0,column=0)
        tk.OptionMenu(
            self.frm_add_agent,
            self.agent_type,
            *agent_types,
            command=partial(self.update_agent_menu)
        ).grid(row=0,column=1)
        self.update_agent_menu(self.agent_type.get())
        tk.Button(
            master=self.frm_add_agent, text="Add Agent",
            command=self.add_agent,
        ).grid(row=1,column=0,sticky="nw")
        self.frm_add_agent.grid(row=4,column=0)
        #show the agents we have added so far
        self.current_agents = []
        self.frm_current_agents = tk.Frame(self, relief=tk.RAISED, bd=2)
        self.frm_current_agents.grid(row=5, column=0)
        #buttons to reset agents, or allow training for ones that don't exist
        tk.Button(
            self,
            text="Reset agents", command=self.reset_agents
        ).grid(row=6, column=0,sticky="nw")
        frm_training = tk.Frame(self)
        tk.Label(frm_training, text="Allow Training").grid(row=0, column=0)
        self.allow_training = tk.BooleanVar(value=False)
        tk.Checkbutton(frm_training, variable=self.allow_training).grid(row=0,column=1)
        frm_training.grid(row=7,column=0)
        #button to start simulation
        tk.Button(self, text="Start Simulation", command=self.start_simulation)


    def add_agent(self):
        """
        Add an agent to the collection of agents to be simulated
        """
        agent_pos = len(self.current_agents)
        #create the frame for displaying info
        frm_agent_info = tk.Frame(self.frm_current_agents, relief=tk.RAISED, bd=2, padx=5, pady=5)
        #fill in the frame
        agent_type = self.agent_type.get()
        tk.Label(frm_agent_info, text=agent_type).pack()
        match agent_type:
            case "General":
                pass
            case "Belief":
                #add depths and beliefs to frame
                tk.Label(
                    frm_agent_info,
                    text=f"Beliefs = {self.agent_settings["beliefs"].get()}"
                ).pack()
                tk.Label(
                    frm_agent_info,
                    text=f"Depth = {self.agent_settings["depth"].get()}"
                ).pack()
            case "Bayesian":
                #add parameters into frame
                tk.Label(frm_agent_info, text=f"Width 1 = {self.agent_settings["width_1"].get()}").pack()
                tk.Label(
                    frm_agent_info,
                    text=f"Width 2 VC = {self.agent_settings["width_2_vc"].get()}"
                ).pack()
                tk.Label(
                    frm_agent_info,
                    text=f"Width 2 Semi VC = {self.agent_settings["width_2_semi_vc"].get()}"
                ).pack()
                tk.Label(
                    frm_agent_info,
                    text=f"Fake Boards = {self.agent_settings["fake_boards"].get()}"
                ).pack()
                if self.sim_type != "Offline":
                    learning = "Online" if self.agent_settings["learning"].get() else "Offline"
                    tk.Label(frm_agent_info, text=learning).pack()
            case "RL":
                pass
            case "From File":
                pass
        frm_agent_info.grid(row=0, column=agent_pos, sticky="ns")
        self.current_agents.append(self.agent_settings.copy())
        self.agent_settings = {}
        self.update_agent_menu(agent_type)


    def reset_agents(self):
        """
        Reset the agents stored in the collection
        """


    def start_simulation(self):
        """
        Runs the simulation on the current agents
        """


    def update_agent_menu(self, agent_type: str):
        """
        Update the settings available in the agent menu based on type
        """
        settings = self.agent_settings
        if self.frm_agent_settings is not None:
            #reset the previous game settings
            for widget in self.frm_agent_settings.winfo_children():
                widget.destroy()
            self.frm_agent_settings.grid_remove()
        frm = tk.Frame(self.frm_add_agent)
        #update menu based on selection
        match agent_type:
            case "General":
                pass
            case "Belief":
                #depth changing
                settings["depth"] = tk.IntVar(value=3)
                tk.Label(frm, text="Depth").pack()
                tk.Entry(frm, textvariable=settings["depth"]).pack()
                #beliefs changing
                settings["beliefs"] = tk.IntVar(value=20)
                tk.Label(frm, text="Beliefs").pack()
                tk.Entry(frm, textvariable=settings["beliefs"]).pack()
            case "Bayesian":
                #width_1 changing
                settings["width_1"] = tk.IntVar(value=5)
                tk.Label(frm, text="Width 1").pack()
                tk.Entry(frm, textvariable=settings["width_1"]).pack()
                #width_2 vc changing
                settings["width_2_vc"] = tk.IntVar(value=3)
                tk.Label(frm, text="Width 2 VC").pack()
                tk.Entry(frm, textvariable=settings["width_2_vc"]).pack()
                #width 2 semi vc changing
                settings["width_2_semi_vc"] = tk.IntVar(value=2)
                tk.Label(frm, text="Width 2 Semi VC").pack()
                tk.Entry(frm, textvariable=settings["width_2_semi_vc"]).pack()
                #width 2 semi vc changing
                settings["fake_boards"] = tk.IntVar(value=10)
                tk.Label(frm, text="Fake Boards").pack()
                tk.Entry(frm, textvariable=settings["fake_boards"]).pack()
                if self.sim_type.get() != "Offline":
                    #add learning parameter
                    settings["learning"] = tk.BooleanVar(value=False)
                    tk.Label(frm, text="Online").pack()
                    tk.Checkbutton(frm, variable=settings["learning"]).pack()
            case "From File":
                pass
        frm.grid(row=1,column=1)
        self.frm_agent_settings = frm


    def draw_graph(self):
        """
        Draw a graph of results following a simulation involving an online agent.
        """


    def draw_table(self):
        """
        Draw a table of results following a simulation involving offline agents.
        """
