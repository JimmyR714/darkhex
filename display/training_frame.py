"""
Module for the training frame where we can train and simulate agents
"""

import tkinter as tk
from tkinter import font
import logging
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import game.util as util


class TrainingFrame(tk.Frame):
    """
    Frame where we can initiate simulations between agents, and display the results.
    """
    def __init__(self, master):
        super().__init__(master=master, relief=tk.RAISED, bd=2)
        #add title
        tk.Label(master=self, text="Training Frame").grid(row=0, column=0, sticky="nw")
        #add game settings
        frm_game_settings = tk.Frame(self)
        #add rows and columns
        self.num_rows = tk.StringVar(value="3")
        tk.Label(frm_game_settings, text="Rows").grid(row=0, column=0)
        tk.Entry(frm_game_settings, textvariable=self.num_rows).grid(row=0, column=1)
        self.num_cols = tk.StringVar(value="3")
        tk.Label(frm_game_settings, text="Cols").grid(row=1, column=0)
        tk.Entry(frm_game_settings, textvariable=self.num_cols).grid(row=1, column=1)
        #add simulation iteration variable
        self.simulation_iters = tk.StringVar(value="25")
        tk.Label(frm_game_settings, text="Simulation Iterations").grid(row=2, column=0)
        tk.Entry(frm_game_settings, textvariable=self.simulation_iters).grid(row=2, column=1)
        #add simulation step variable
        self.simulation_step = tk.StringVar(value="5")
        tk.Label(frm_game_settings, text="Iteration Step").grid(row=3, column=0)
        tk.Entry(frm_game_settings, textvariable=self.simulation_step).grid(row=3, column=1)
        #add simulation option selection
        sim_options = [
            "Offline",
            "Online vs Offline",
            "Online"
        ]
        tk.Label(frm_game_settings, text="Simulation Type").grid(row=4, column=0)
        self.sim_type = tk.StringVar(value="Offline")
        tk.OptionMenu(
            frm_game_settings,
            self.sim_type,
            *sim_options,
            #command=partial(self.update_agent_and_game_menu, 1)
        ).grid(row=4, column=1)
        frm_game_settings.grid(row=1, column=0, sticky="nw")
        #allow us to add an agent to list of agents to be run
        self.frm_add_agent = tk.Frame(master=self)
        #add agent type selection
        agent_types = [
            "General",
            "Belief",
            "RL",
            "Bayesian",
            "From File"
        ]
        self.agent_type = tk.StringVar(value="General")
        tk.Label(master=self.frm_add_agent, text="Select Agent Type:").grid(row=0,column=0)
        tk.OptionMenu(
            self.frm_add_agent,
            self.agent_type,
            *agent_types,
            command=partial(self.update_agent_menu)
        ).grid(row=0,column=1)
        #add agent settings depending on agent type
        self.agent_settings = {}
        self.frm_agent_settings = None
        self.update_agent_menu(self.agent_type.get())
        #add agent naming
        self.agent_name = tk.StringVar(value="Agent Name")
        tk.Entry(self.frm_add_agent, textvariable=self.agent_name).grid(row=1,column=0)
        #allow user to add agent to collection
        tk.Button(
            master=self.frm_add_agent, text="Add Agent",
            command=self.add_agent,
        ).grid(row=1,column=1)
        self.frm_add_agent.grid(row=2,column=0,sticky="nw")
        #show the agents we have added so far
        self.current_agents = []
        self.online_agents = []
        self.frm_current_agents = tk.Frame(self, relief=tk.RAISED, bd=2, padx=5, pady=5)
        self.frm_current_agents.grid(row=3, column=0,sticky="nw")
        #buttons to reset agents, or allow training for ones that don't exist
        tk.Button(
            self,
            text="Reset agents", command=self.reset_agents
        ).grid(row=4, column=0,sticky="nw")
        frm_training = tk.Frame(self)
        tk.Label(frm_training, text="Allow Training").grid(row=0, column=0)
        self.allow_training = tk.BooleanVar(value=False)
        tk.Checkbutton(frm_training, variable=self.allow_training).grid(row=0,column=1)
        tk.Label(frm_training, text="Training Iterations").grid(row=0, column=2)
        self.training_iters = tk.StringVar(value="10")
        tk.Entry(frm_training, textvariable=self.training_iters).grid(row=0, column=3)
        frm_training.grid(row=5,column=0,sticky="nw")
        #button to start simulation
        tk.Button(
            self,
            text="Start Simulation", command=self.start_simulation
        ).grid(row=6,column=0,sticky="nw")


    def add_agent(self):
        """
        Add an agent to the collection of agents to be simulated
        """
        agent_pos = len(self.current_agents)
        #create the frame for displaying info
        frm_agent_info = tk.Frame(self.frm_current_agents, relief=tk.RAISED, bd=2, padx=5, pady=5)
        #fill in the frame
        add = True
        agent_type = self.agent_type.get()
        agent_name = self.agent_name.get()
        self.agent_settings["type"] = agent_type
        self.agent_settings["name"] = agent_name
        tk.Label(frm_agent_info, text=agent_name,font=font.Font(underline=True)).pack()
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
                tk.Label(
                    frm_agent_info,
                    text=f"Width 1 = {self.agent_settings["width_1"].get()}"
                ).pack()
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
                if self.sim_type.get() != "Offline":
                    learning_bool = self.agent_settings["learning"].get()
                    learning = "Online" if learning_bool else "Offline"
                    tk.Label(frm_agent_info, text=learning).pack()
                    add = False
                    self.online_agents.append(self.agent_settings.copy())
            case "RL":
                pass
            case "From File":
                pass
        frm_agent_info.grid(row=0, column=agent_pos, sticky="ns")
        if add:
            self.current_agents.append(self.agent_settings.copy())
        self.agent_settings = {}
        self.update_agent_menu(agent_type)


    def reset_agents(self):
        """
        Reset the agents stored in the collection
        """
        for widget in self.frm_current_agents.winfo_children():
            widget.destroy()
        self.current_agents = []


    def start_simulation(self):
        """
        Runs the simulation on the current agents
        """
        #don't run when no agents have been selected
        if len(self.current_agents) == 0:
            return
        match self.sim_type.get():
            case "Offline":
                #get all combinations of agents
                agent_combs = [
                    (agent_1, agent_2)
                    for agent_1 in self.current_agents
                    for agent_2 in self.current_agents
                ]
                #simulate every combination and save results
                results = pd.DataFrame(
                    columns=[a["name"] for a in self.current_agents],
                    index=[a["name"] for a in self.current_agents]
                )
                for agent_1, agent_2 in agent_combs:
                    agent_1["colour"] = "w"
                    agent_2["colour"] = "b"
                    iterations = int(self.simulation_iters.get())
                    #run the new game in the controller
                    agent_1_name = agent_1["name"]
                    agent_2_name = agent_2["name"]
                    logging.info("Simulating match: %s vs %s", agent_1_name, agent_2_name)
                    self.master.controller.new_game(
                        num_cols=int(self.num_cols.get()),
                        num_rows=int(self.num_rows.get())
                    )
                    wins = self.master.controller.new_ava_game(
                        agent_1, agent_2, iterations
                    )
                    results.at[agent_1_name, agent_2_name] = wins
                self.draw_table(results)
            case "Online vs Offline":
                assert len(self.online_agents) > 0
                iters = int(self.simulation_iters.get())
                step = self.simulation_step.get()
                iter_range = range(step, iters, step)
                if len(self.online_agents) == 1:
                    #if there is only one online agent, we put all offline agents on the same graph
                    #simulate online vs each offline agent for each colour
                    online_agent = self.online_agents[0]
                    online_agent["colour"] = "w"
                    #for each colour matchup
                    for offline_colour in ["b", "w"]:
                        #we create a new results frame for the graph
                        results = pd.DataFrame(
                            columns=[a["name"] for a in self.current_agents],
                            index=iter_range
                        )
                        #for every offline agent
                        for offline_agent in self.current_agents:
                            #we simulate the game
                            offline_agent["colour"] = offline_colour
                            offline_agent_name = offline_agent["name"]
                            self.master.controller.new_game(
                                num_cols=int(self.num_cols.get()),
                                num_rows=int(self.num_rows.get())
                            )
                            #simulate each agent with the correct colour
                            if offline_colour == "w":
                                wins = self.master.controller.new_ava_game(
                                    offline_agent, online_agent, iters
                                )
                            else:
                                wins = self.master.controller.new_ava_game(
                                    online_agent, offline_agent, iters
                                )
                            #then add to our results table
                            for i, num_iters in enumerate(iter_range):
                                results.at[num_iters, offline_agent_name] = wins[i]
                        #when we are done with one colour, we plot the graph and switch
                        self.draw_graph(results, title=f"Online {online_agent["colour"]}")
                        online_agent["colour"] = "b"
                else:
                    #if there is more than one online agent,
                    #we produce a graph for each offline agent,
                    #showing online performance vs a single agent

                    #for each offline agent (i.e. each graph)
                    for offline_agent in self.current_agents:
                        #we get a new set of results
                        results = pd.DataFrame(
                            columns=[
                                (agent["name"], colour)
                                for agent in self.online_agents
                                for colour in ["w", "b"]
                            ],
                            index=iter_range
                        )
                        offline_agent_name = offline_agent["name"]
                        #for each colour and online agent pair
                        for offline_colour in ["b", "w"]:
                            offline_agent["colour"] = offline_colour
                            for online_agent in self.online_agents:
                                online_agent["colour"] = util.swap_colour(offline_colour)
                                online_agent_name = online_agent["name"]
                                #we simulate the game
                                self.master.controller.new_game(
                                    num_cols=int(self.num_cols.get()),
                                    num_rows=int(self.num_rows.get())
                                )
                                #ensure each agent is the correct colour
                                if offline_colour == "w":
                                    wins = self.master.controller.new_ava_game(
                                        offline_agent, online_agent, iters
                                    )
                                else:
                                    wins = self.master.controller.new_ava_game(
                                        online_agent, offline_agent, iters
                                    )
                                #input values into the results table
                                for i, num_iters in enumerate(iter_range):
                                    results.at[
                                        num_iters,
                                        (online_agent_name, online_agent["colour"])
                                    ] = wins[i]
                        self.draw_graph(results, title=f"{offline_agent_name}")
            case "Online":
                #we only have one combination
                assert len(self.current_agents) == 0
                assert len(self.online_agents) == 2


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
                settings["learning"] = tk.BooleanVar(value=False)
                if self.sim_type.get() != "Offline":
                    #add learning parameter
                    tk.Label(frm, text="Online").pack()
                    tk.Checkbutton(frm, variable=settings["learning"]).pack()
            case "From File":
                pass
        frm.grid(row=2,column=0)
        self.frm_agent_settings = frm


    def draw_graph(self, results: pd.DataFrame, title: str):
        """
        Draw a graph of results following a simulation involving an online agent.
        """
        #save the graph to a file
        results.plot(title=title)
        plt.savefig(f"simulations\\online\\plot_{title}.png")
        #also save data to the file
        results.to_csv(f"simulations\\online\\{title}.csv", encoding="utf-8")



    def draw_table(self, results: pd.DataFrame):
        """
        Draw a table of results following a simulation involving offline agents.
        """
        print(results.to_latex())
        #also save the table to a file
        results.to_csv("simulations\\results.csv", encoding="utf-8")
