"""
Module that includes an agent that has an abstracted perfect view of the board,
The imperfect aspect is included using a belief system that the agent never sees.
"""

import os
from pprint import pprint
from pathlib import Path
import math
import logging
import json
import itertools
from functools import partial
from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD
import pyro.distributions as distribution
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule
import torch
import pandas as pd
import gymnasium as gym
import numpy as np
from agents.agent import Agent
import game.util as util
import game.hex


LEARNING_SIZE = 1000


class AbstractAgent(Agent):
    """
    Agent that utilizes an advanced Hex AI and belief states
    To play Dark Hex
    """
    def __init__(self, num_cols: int, num_rows: int, settings: dict, hex_path: str):
        """
        Create an abstract agent
        """
        super().__init__(num_cols, num_rows, settings)
        # assign all settings here
        # cell cpds are used to give values to parents in a network
        self.cell_cpds : dict[tuple[int,int], FunctionalCPD] = {}
        logging.info("Setting cell cpds")
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0),
            update=False
        )
        logging.info("Cell cpds set")
        self.num_boards = settings["fake_boards"]
        self.weightings : dict[str, int] = settings
        self.learning = settings["learning"]
        # each item in here is a model with the key being the child cell
        self.cell_networks : dict[tuple[int,int], FunctionalBayesianNetwork] = {}
        logging.info("Creating cell networks")
        self.initialise_networks()
        logging.info("Cell networks created.")
        self.opponents_unseen = 0
        self.possible_empty_cells = num_cols * num_rows
        if self.colour == "b":
            self.opponents_unseen += 1
        logging.info("Getting Hex Agent")
        self.hex_agent = HexAgent.from_file(hex_path)
        logging.info("Got Hex Agent.")
        self.int_board = [0]*num_cols*num_rows


    def move(self) -> tuple[int, int]:
        """
        Make a move by the agent. 
        We search for the correct move in the most likely states
        and return the one with the highest expected gain
        """
        # first simulate the bayesian networks to find most likely places of their cells
        cell_count = {}
        logging.info("Simulating cell networks")
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # simulate network for 100 samples
                sim_outputs = self.cell_networks[(col, row)].simulate(n_samples=100)
                cell_count.update({(col,row): sim_outputs[(col, row)].sum()})
        logging.info("Cell networks simulated")
        #sort the cell counts
        decr_counts = sorted(
            [
            (k,v) for k,v in cell_count.items() if self.int_board[k[1] * self.num_cols + k[0]] == 0
            ],
            key=lambda x: x[1],
            reverse=True
        )
        fake_boards = self.fake_boards(decr_counts)
        # run Hex AI on the fake boards
        scores = [0] * self.num_cols * self.num_rows
        for fb in fake_boards:
            logging.info("Running Hex Agent on the fake board:\n%s", fb)
            #update scores
            scores = [sum(x) for x in zip(self.hex_agent.board_scores(fb), scores)]
        #now get the best move
        best_score = -1000 * ((2*int(self.colour == "w"))-1)
        best_move = (0,0)
        for pos, val in enumerate(scores):
            col = pos % self.num_cols
            row = pos // self.num_cols
            if self.board[row][col] == "e":
                good_white = self.colour == "w" and val > best_score
                good_black = self.colour == "b" and val < best_score
                if good_white or good_black:
                    best_score = val
                    best_move = (col+1,row+1)
        return best_move


    def update_information(self, col: int, row: int, colour: str):
        """
        Update our belief system with the new information
        """
        if self.board[row-1][col-1] == "e":
            #if this piece is new to us
            self.possible_empty_cells -= 1
            if colour == self.colour:
                #if it is our piece being placed, the opponent will move somewhere
                self.opponents_unseen += 1
            else:
                #we have found one of the opponent's pieces
                self.opponents_unseen -= 1
        move_again = super().update_information(col,row,colour)
        #adjust col and row
        col-=1
        row-=1
        self.int_board[row*self.num_rows + col] = ((2*int(colour == "w"))-1)
        if colour != self.colour:
            # we know (col, row) contains their colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(1))
            self.fixed_network(col,row,1)
        else:
            # we know (col, row) contains our colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(0))
            self.fixed_network(col,row,0)
        #TODO find the real reason for the cell count bug
        cell_count = min(1, max(0, self.opponents_unseen / self.possible_empty_cells))
        self.set_cell_cpds(
            cells=self._possible_empty_cells(),
            dist=distribution.Bernoulli(cell_count)
        )
        #TODO find a better distribution for the cell cpds
        return move_again


    def reset(self, board: list[list[str]]):
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0),
            update=True
        )
        if self.learning:
            self.learn(board)
        self.int_board = [0]*self.num_cols*self.num_rows
        self.possible_empty_cells = self.num_cols * self.num_rows
        self.opponents_unseen = 0
        if self.colour == "b":
            self.opponents_unseen += 1
        self.board = [["e"]*self.num_cols for i in range(self.num_rows)]


    def set_cell_cpds(self, cells: list[tuple[int, int]], dist: distribution, update = True):
        """
        Set a list of cell cpds to be the same distribution
        """
        for cell in cells:
            if update:
                self.update_cell_cpd(cell[0], cell[1], dist)
            else:
                self.cell_cpds.update({
                    cell: FunctionalCPD(
                        variable=cell,
                        fn=lambda _: dist,
                        parents=[]
                    )
                })


    def update_cell_cpd(self, col: int, row: int, dist: distribution):
        """
        Updates a single cell cpd to be the new distribution. 
        Resets the relevant cell cpd in all models where it is present.
        """
        new_cpd = FunctionalCPD(
            variable=(col,row),
            fn = lambda _: dist,
            parents=[]
            )
        for cell in self._other_cells(col, row):
            self.cell_networks[cell].remove_cpds(self.cell_cpds[(col,row)])
            self.cell_networks[cell].add_cpds(new_cpd)
        self.cell_cpds[(col,row)] = new_cpd


    def fixed_network(self, col: int, row: int, value: int):
        """
        Sets a cell network to be fixed and always return a certain value.
        """
        assert value <= 1
        #a fixed function where the parent has no effect
        def fixed_func(val, _):
            return distribution.Bernoulli(val)
        #create new model
        parents = self._other_cells(col, row)
        network_edges = [(cell, (col,row)) for cell in parents]
        model = FunctionalBayesianNetwork(network_edges)
        #add our fixed cpd
        cpd = FunctionalCPD((col,row), partial(fixed_func, value), parents)
        model.add_cpds(cpd)
        #add cell cpds
        cell_cpds = [self.cell_cpds[cell] for cell in self._other_cells(col, row)]
        for cell_cpd in cell_cpds:
            model.add_cpds(cell_cpd)
        assert model.check_model()
        self.cell_networks[(col,row)] = model


    def initialise_networks(self):
        """
        Create the Bayesian Networks that contain our beliefs.
        Requires that the cell cpds have been set.
        """
        def cell_exists(cell) -> bool:
            """
            Check whether a potential cell exists
            
            Parameters:
                cell: tuple[int, int] - The cell that we are checking the existence of.
            """
            top_left = cell[0] >= 0 and cell[1] >= 0
            bottom_right = cell[0] < self.num_cols and cell[1] < self.num_rows
            return top_left and bottom_right


        def add_cond(cond: str,
                     cells: list[tuple[int, int]]) -> tuple[dict[tuple[int,int], int], int]:
            """
            Adds a condition to the conditional probability distribution we are creating. 
            All elements returned should be combined with the currently maintained values.
            
            Parameters:
                cond: str - The condition we will be adding.
                cells: list[tuple[int, int]] - The cells that the condition includes
                
            Returns:
                cond_parent_weightings: dict[tuple[int,int], int] - 
                The parents used in this condition and their weightings.
                cond_weightings : int - The sum of weightings from this condition.
            """
            cond_weighting = 0
            cond_parent_weightings = {}
            # for each cell valid in this condition
            for cell in cells:
                if cell_exists(cell):
                    # add the cell to the function
                    cond_weighting += self.weightings[cond]
                    cond_parent_weightings.update({cell: self.weightings[cond]})
            return (cond_parent_weightings, cond_weighting)


        #define weight function for creating a cpd
        def weight_dist(weightings: dict[tuple[int,int], int], total_weighting: int, parent):
            #for now, we use a beta distribution with
            # alpha = (weightings * present values)^2
            # beta = total weighting
            total = 1
            for item in weightings.items():
                total += parent[item[0]] * item[1]
            return distribution.Beta(total**2, total_weighting)


        # note here top left playable cell is (0,0), (col, row)

        conditions = [
            "width_1", "width_2_vc", "width_2_semi_vc"
        ]
        valid_cells = {
            "width_1": lambda col, row: [
                (col-1, row), (col, row-1), (col+1, row-1),
                (col+1, row), (col, row+1), (col-1,row+1)
            ],
            "width_2_vc": lambda col, row: [
                (col+1, row-2), (col+2, row-1), (col+1, row+1),
                (col-1, row+1), (col-2, row+1), (col-1, row-1)
            ],
            "width_2_semi_vc": lambda col, row: [
                (col+2, row-1), (col+2, row), (col, row+2),
                (col-2, row+2), (col-2, row), (col, row-2)
            ]
        }

        assert self.cell_cpds is not None

        # create a network centered around each node
        cell_networks = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # first create the cpd for this cell
                total_weighting = 0
                parent_weightings = {}
                #start with all parent weightings being 0
                for cell in self._other_cells(col, row):
                    parent_weightings.update({cell:0})
                # determine distribution based on some conditions
                for cond in conditions:
                    if self.weightings[cond] > 0:
                        # add weightings of all cells satisfying the condition
                        new_parent_weightings, new_weighting = add_cond(
                            cond = cond,
                            cells = valid_cells[cond](col, row)
                        )
                        #update the parent weightings
                        for item in new_parent_weightings.items():
                            parent_weightings.update({
                                item[0]: item[1] + new_parent_weightings[item[0]]
                            })
                        total_weighting += new_weighting
                #create cpd
                cpd = FunctionalCPD(
                    variable=(col, row),
                    fn=partial(weight_dist, parent_weightings, total_weighting),
                    parents=self._other_cells(col,row)
                )
                #now we create the network for this cell
                network_edges = [(cell, (col,row)) for cell in self._other_cells(col, row)]
                model = FunctionalBayesianNetwork(network_edges)
                model.add_cpds(cpd)
                cell_cpds = [self.cell_cpds[cell] for cell in self._other_cells(col, row)]
                for cell_cpd in cell_cpds:
                    model.add_cpds(cell_cpd)
                assert model.check_model()
                cell_networks.update({(col,row): model})
        self.cell_networks = cell_networks


    def fake_boards(self, sorted_counts: list[tuple[tuple[int,int], float]]) -> list[list[int]]:
        """
        Creates multiple fake boards.
        """
        extra_cells = len(sorted_counts) - self.opponents_unseen
        only_cells = [k for (k,v) in sorted_counts]
        if extra_cells == 0:
            fake_boards = [self.fake_board(only_cells[:self.opponents_unseen])]
        else:
            #learn a fixed number of most likely combinations
            fake_boards = []
            all_combs = list(itertools.combinations(only_cells, self.opponents_unseen))
            for comb in all_combs[:min(len(all_combs), self.num_boards)]:
                fake_boards.append(self.fake_board(comb))
        return fake_boards


    def fake_board(self, cells: list[tuple[int, int]]) -> list[int]:
        """
        Creates a fake board based on what we know and extra guessed cells.
        """
        new_board = self.int_board.copy()
        cell_value = -1 * ((2*int(self.colour == "w"))-1)
        for cell in cells:
            new_board[cell[1]* self.num_cols + cell[0]] = cell_value
        return new_board


    def learn(self, board: list[list[str]]):
        """
        Use the final board state to update our cpds.
        Note that the input board includes the borders.
        """
        # get opponent's pieces into a dictionary
        final_states = []
        for y, row in enumerate(board[1:self.num_rows+1]):
            for x, cell in enumerate(row[1:self.num_cols+1]):
                if cell == util.swap_colour(self.colour):
                    final_states.append((x,y))
        # our cell data is {(col,row) : table of data points,
        # each point corresponds to points in other cell tables}
        logging.debug("Creating cell data")
        all_cells = self._other_cells(-1,-1)
        rows_list = []
        #learn the possible positions for every combination at each stage of the game
        for i in range(1,len(final_states)):
            #TODO allow adjustable learning size
            if math.comb(len(final_states), i) < LEARNING_SIZE:
                # we only learn combinations if the size is small
                combs = list(itertools.combinations(final_states, i))
                for comb in combs:
                    #create a datapoint for every combination
                    rows_list.append({cell:int(cell in comb) for cell in all_cells})
        # learn the data for each model
        cell_data = pd.DataFrame(rows_list)
        logging.info("Learning cell data for each network")
        for network in self.cell_networks.values():
            network.fit(cell_data)


    def _possible_empty_cells(self) -> list[tuple[int, int]]:
        """
        Return a list of the possible empty cells on the board
        """
        return [
            (x,y)
            for x in range(self.num_cols)
            for y in range(self.num_rows)
            if self.board[y][x] == "e"
        ]


    def _other_cells(self, col: int, row: int) -> list[tuple[int, int]]:
        """
        Return a list of all cells other than the input one
        """
        return [
            (x,y)
            for x in range(self.num_cols)
            for y in range(self.num_rows)
            if x!=col or y!=row
        ]


# For the hex AI, we require some way of returning value of the move,
# ideally the value of multiple moves to add to a large expected value
# accumulation, so we can calculate the true best expected value of a move

class HexAgent():
    """
    Class for the base Hex AI that the abstracted agent uses.
    Very similar to the Dark Hex rl agent.
    """
    def __init__(self, num_cols: int, num_rows: int, agent_colour: str,
                 rl_module: RLModule, algo: Algorithm = None):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.colour = agent_colour
        self.env = HexEnv1(config={
                    "num_cols": num_cols,
                    "num_rows": num_rows
                })
        self.algo = algo
        self.rl_module = rl_module


    @classmethod
    def from_file(cls, path: str):
        """
        Retrieve a Hex agent from a file path.
        The agent colour and number of cols and rows must also be stored in that path
        """
        with open(os.path.join(path, "settings.json"), "r", encoding="utf-8") as f:
            settings = json.load(f)
        rl_module = RLModule.from_checkpoint(
            Path(path)
            / "learner_group"
            / "learner"
            / "rl_module"
            / settings["colour"]
        )
        return cls(
            num_cols=settings["num_cols"],
            num_rows=settings["num_rows"],
            agent_colour=settings["colour"],
            rl_module=rl_module
        )


    @classmethod
    def to_train(cls, num_cols: int, num_rows: int, colour: str):
        """
        Create a fresh, untrained agent.
        """
        config = (
            PPOConfig()
            .env_runners(
                env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
            )
            .environment(
                HexEnv,
                env_config = {
                    "num_cols": num_cols,
                    "num_rows": num_rows
                }
            )
            .multi_agent(
                policies={"w", "b"},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: (
                    agent_id
                )
            )
            .training(
                gamma=0.8,
                lr=0.01,
                kl_coeff=0.2,
                vf_loss_coeff=0.01,
                use_kl_loss=True,
                clip_param=0.1
            )
            .rl_module(
                model_config=DefaultModelConfig(
                    use_lstm=False,
                    fcnet_hiddens=[256, 256],
                    max_seq_len=15,
                    vf_share_layers=True,
                ),
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "w": RLModuleSpec(),
                        "b": RLModuleSpec(),
                    }
                ),
            )
        )
        return cls(
            num_cols=num_cols,
            num_rows=num_rows,
            agent_colour=colour,
            rl_module=None,
            algo=config.build_algo()
        )


    def train(self, iterations = 5):
        """
        Train the algorithm and print the results
        """
        # Train it for some number of iterations
        for _ in range(iterations):
            pprint(self.algo.train())


    def save(self, path: str):
        """
        Save the algorithm to a path. 
        It can then be used in future to play the game.
        """
        self.algo.save_to_path(path)
        with open(os.path.join(path, "settings.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "num_cols": self.num_cols,
                "num_rows": self.num_rows,
                "colour": self.colour
            }, indent=4))


    def board_scores(self, board: list[int]) -> list[float]:
        """
        Find the expected values of playing in each cell within the board.
        Returns:
            board_scores: list[list[float]] - The scores of every cell on the board.
        """
        #set the board in the environment
        self.env.set_board(board, self.colour)
        #compute the action distribution
        obs_batch = torch.from_numpy(np.array(board, np.float32)).unsqueeze(0).float()
        model_outputs = self.rl_module.forward_inference({"obs": obs_batch})
        action_dist = model_outputs["action_dist_inputs"][0].numpy()
        return action_dist


class HexEnv(MultiAgentEnv):
    """
    Environment for Hex.
    To be passed into PPO algorithm.
    Mostly copied from the Dark Hex one.
    """
    def __init__(self, config : dict[str, int]):
        super().__init__()
        num_cols : int = config["num_cols"]
        num_rows : int = config["num_rows"]
        #define the agents in the game
        self.agents = self.possible_agents = ["w", "b"]

        #each agent observes a space of size num_cols x num_rows
        #each cell can range from -1.0 to 1.0, with -1.0 being black and 1.0 being white
        self.observation_spaces = {
            "w": gym.spaces.Box(-1.0, 1.0, (num_cols* num_rows,), np.float32),
            "b": gym.spaces.Box(-1.0, 1.0, (num_cols* num_rows,), np.float32)
        }

        #each agent can place in any of the cells
        self.action_spaces = {
            "w": gym.spaces.Discrete(num_cols * num_rows),
            "b": gym.spaces.Discrete(num_cols * num_rows)
        }

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.abstract_game = game.hex.AbstractHex(self.num_cols, self.num_rows)
        self.board = [0] * self.num_cols * self.num_rows


    def reset(self, *, seed=None, options=None):
        """
        Initialise the abstract game and set the starting players
        """
        #create abstract game of correct size
        self.abstract_game = game.hex.AbstractHex(self.num_cols, self.num_rows)
        self.board = [0] * self.num_cols * self.num_rows
        # return observation dict and infos dict.
        return {"w": np.array(self.board, np.float32)}, {}


    def step(self, action_dict):
        """
        Do one step in this episode
        """
        #action is (col-1 * num_rows) + row-1
        initial_turn = self.abstract_game.turn
        action : int = action_dict[initial_turn]
        col = (action // self.num_cols) + 1
        row = (action % self.num_cols) + 1
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {initial_turn: 0.0}
        # Create a terminated-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminated = {"__all__": False}
        #do the move
        #we still use the dark hex abstract game because it works in the same way
        move_result = self.abstract_game.move(col=col, row=row, colour=initial_turn)
        #get rewards, termination, and board updates of the move
        #if move_result in ["full_white", "full_black"] or (
        #    move_result == "placed" and self.board[action] != 0):
        #    #penalize trying to place a piece in an occupied field
        #    rewards[initial_turn] -= 5.0
        #    rewards[util.swap_colour(initial_turn)] = 5.0
        if move_result in ["white_win", "black_win"] or (
            move_result=="placed" and self.board[action] != (2*int(initial_turn == "w"))-1):
            #place the piece
            self.board[action] = 1 if initial_turn == "w" else -1
            if move_result == util.colour_map[initial_turn]+"_win":
                rewards[initial_turn] = 10.0
                rewards[util.swap_colour(initial_turn)] = -10.0
                terminated["__all__"] = True
        new_turn = self.abstract_game.turn
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return (
            {
                new_turn: np.array(self.board, np.float32),
            },
            rewards,
            terminated,
            {},
            {}
        )


    def set_board(self, board: list[int], agent_colour: str):
        """
        Set the abstract game to be a certain fake board.
        """
        self.abstract_game.set_board(board, agent_colour)


#temp class for hex
class HexEnv1(MultiAgentEnv):
    """
    Environment for Dark Hex.
    To be passed into PPO algorithm.
    """
    def __init__(self, config : dict[str, int]):
        super().__init__()
        num_cols : int = config["num_cols"]
        num_rows : int = config["num_rows"]
        #define the agents in the game
        self.agents = self.possible_agents = ["w", "b"]

        #each agent observes a space of size num_cols x num_rows
        #each cell can range from -1.0 to 1.0, with -1.0 being black and 1.0 being white
        self.observation_spaces = {
            "w": gym.spaces.Box(-1.0, 1.0, (num_cols* num_rows,), np.float32),
            "b": gym.spaces.Box(-1.0, 1.0, (num_cols* num_rows,), np.float32)
        }

        #each agent can place in any of the cells
        self.action_spaces = {
            "w": gym.spaces.Discrete(num_cols * num_rows),
            "b": gym.spaces.Discrete(num_cols * num_rows)
        }

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.abstract_game = None
        self.white_board = None
        self.black_board = None


    def reset(self, *, seed=None, options=None):
        """
        Initialise the abstract game and set the starting players
        """
        #create abstract game of correct size
        self.abstract_game = game.hex.AbstractHex(self.num_cols, self.num_rows)
        self.white_board = [0] * self.num_cols * self.num_rows
        self.black_board = [0] * self.num_cols * self.num_rows
        # return observation dict and infos dict.
        return {"w": np.array(self.white_board, np.float32)}, {}


    def step(self, action_dict):
        """
        Do one step in this episode
        """
        if "w" in action_dict:
            action : int = action_dict["w"]
            initial_turn = "w"
        else:
            action : int = action_dict["b"]
            initial_turn = "b"
        #action is (col-1 * num_rows) + row-1
        #action : int = action_dict[self.abstract_game.turn]
        col = (action // self.num_cols) + 1
        row = (action % self.num_cols) + 1
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {"w": 0.0, "b": 0.0}
        # Create a terminated-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminated = {"__all__": False}
        #do the move
        move_result = self.abstract_game.move(col=col, row=row, colour=initial_turn)
        #get rewards, termination, and board updates of the move
        logging.debug("Move result in step is: %s", move_result)
        match move_result:
            case "black_win":
                self.black_board[action] = -1
                self.white_board[action] = -1
                rewards["b"] += 10.0
                rewards["w"] -= 10.0
                terminated["__all__"] = True
            case "white_win":
                self.white_board[action] = 1
                self.black_board[action] = 1
                rewards["w"] += 10.0
                rewards["b"] -= 10.0
                terminated["__all__"] = True
            case "full_white":
                self.black_board[action] = 1
                self.white_board[action] = 1
            case "full_black":
                self.white_board[action] = -1
                self.black_board[action] = -1
            case "placed":
                # hugely penalize placing a piece in an occupied cell,
                # hence the algorithm shouldn't ever choose to do it
                cell_value = (2*int(initial_turn == "w"))-1
                if self.get_board(initial_turn)[action] == cell_value:
                    #we have played here before
                    rewards[initial_turn] -= 100.0
                    rewards[util.swap_colour(initial_turn)] += 100.0
                else:
                    self.white_board[action] = cell_value
                    self.black_board[action] = cell_value

        turn = self.abstract_game.turn
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return (
            self.get_obs(turn),
            rewards,
            terminated,
            {},
            {}
        )


    def update_board(self, colour: str, position: int, value: int):
        """
        WARNING: Be careful with use, most updates should use the step function.
        Does a crude update of the board of a given colour in a position.
        """
        self.get_board(colour)[position] = value


    def get_obs(self, colour:str) -> dict:
        """
        Return the current observations of a given colour
        """
        return {colour: np.array(self.get_board(colour), np.float32)}


    def get_board(self, colour: str) -> list[int]:
        """
        Returns the board for the given colour
        """
        if colour == "w":
            return self.white_board
        else:
            return self.black_board


    def set_board(self, board: list[int], agent_colour: str):
        """
        Set the abstract game to be a certain fake board.
        """
        if self.abstract_game is not None:
            self.abstract_game.set_board(board, agent_colour)
            self.white_board = board.copy()
            self.black_board = board.copy()


#run to train the agent
def main():
    """
    Train the agent on a certain board size
    """
    num_cols = 3
    num_rows = 3
    colour = "w"
    logging.info("Creating Agent")
    agent = HexAgent.to_train(num_cols=num_cols, num_rows=num_rows, colour=colour)
    logging.info("Training Agent")
    agent.train(iterations=100)
    logging.info("Agent trained")
    agent.save(
        os.path.join(os.path.dirname(__file__), "trained_agents\\hex_agent_" + str(
            num_cols) + "x" + str(num_rows) + "_" + colour)
    )


if __name__ == "__main__":
    main()
