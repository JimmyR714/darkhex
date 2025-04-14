"""
Module that includes an agent that has an abstracted perfect view of the board,
The imperfect aspect is included using a belief system that the agent never sees.
"""

import os
from pprint import pprint
from pathlib import Path
import logging
import json
from collections.abc import Callable
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
import gymnasium as gym
import numpy as np
from agents.agent import Agent
import game.util as util
import game.darkhex as dh

class AbstractAgent(Agent):
    """
    Agent that utilizes an advanced Hex AI and belief states
    To play Dark Hex
    """
    def __init__(self, num_cols: int, num_rows: int, settings: dict):
        """
        Create an abstract agent
        """
        super().__init__(num_cols, num_rows, settings)
        # assign all settings here
        self.cell_cpds : dict[tuple[int,int], FunctionalCPD] = None
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0)
        )
        self.cell_networks : dict[tuple[int,int], FunctionalBayesianNetwork] = None
        self.initialise_networks()
        self.weightings : dict[str, int] = settings["weightings"]
        self.opponents_unseen = 0
        self.empty_cells = num_cols * num_rows


    def move(self):
        """
        Make a move by the agent. 
        We search for the correct move in the most likely states
        and return the one with the highest expected gain
        """
        # first simulate the bayesian networks to find most likely places of their cells
        cell_count = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # simulate network for 100 samples
                sim_outputs = self.cell_networks[(col, row)].simulate(n_samples=100)
                cell_count.update({(col,row): (sim_outputs[(col, row)] == 1).sum()})
        # for now, we just place cells in the locations that were most common
        # TODO further rounds of simulation?
        guessed_cells = []
        for _ in range(self.opponents_unseen):
            max_cell = max(cell_count, key=cell_count.get)
            guessed_cells.append(max_cell)
            del cell_count[max_cell]
        fake_board = self.fake_board(guessed_cells)
        # run Hex AI on the fake board


    def update_information(self, col: int, row: int, colour: str):
        """
        Update our belief system with the new information
        """
        move_again = super().update_information(col, row, colour)
        if colour != self.colour:
            # we know (col, row) contains their colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(1))
            # we found one of their cells
            self.opponents_unseen -= 1
        else:
            # we know (col, row) contains our colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(0))
            # the opponent will play somewhere that we haven't seen it
            self.opponents_unseen += 1
            self.empty_cells -= 2
        self.set_cell_cpds(
            cells=self._empty_cells(),
            dist=distribution.Bernoulli(self.opponents_unseen / self.empty_cells)
        )
        #TODO find a better distribution for the cell cpds
        return move_again


    def reset(self):
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0)
        )
        self.initialise_networks()
        return super().reset()


    def set_cell_cpds(self, cells: list[tuple[int, int]], dist: distribution):
        """
        Set a list of cell cpds to be the same distribution
        """
        for cell in cells:
            self.update_cell_cpd(cell[0], cell[1], dist)


    def update_cell_cpd(self, col: int, row: int, dist: distribution):
        """
        Updates a single cell cpd to be the new distribution. 
        Resets the relevant cell cpd in all models it is present.
        """
        new_cpd = FunctionalCPD((col,row), dist)
        for cell in self._other_cells(col, row):
            self.cell_networks[cell].remove_cpds(self.cell_cpds[(col,row)])
            self.cell_networks[cell].add_cpds(new_cpd)
        self.cell_cpds[(col,row)] = new_cpd


    def initialise_networks(self):
        """
        Create the Bayesian Networks that contain our beliefs.
        Requires that the cell cpds have been set
        """
        def cell_exists(cell) -> bool:
            """
            Check whether a potential cell exists
            
            Parameters:
                cell: tuple[int, int] - The cell that we are checking the existence of.
            """
            return cell[0] >= 0 and cell[1] >= 0 and cell[0] < self.num_cols and cell[1] < self.num_rows


        def add_cond(cond: str, cells: list[tuple[int, int]]) -> tuple[(Callable), set[int], int]:
            """
            Adds a condition to the conditional probability distribution we are creating. 
            All elements returned should be combined with the currently maintained values.
            
            Parameters:
                cond: str - The condition we will be adding.
                cells: list[tuple[int, int]] - The cells that the condition includes
                
            Returns:
                fn: function[dict, int] - The function after adding the condition.
                parents: set[int] - The parents used in this condition.
                cond_weightings : int - The sum of weightings from this condition.
            """
            cond_weighting = 0
            parents = set()
            fn = lambda parent: 0
            # for each cell valid in this condition
            for cell in cells:
                if cell_exists(cell):
                    # add the cell to the function
                    cond_weighting += self.weightings[cond]
                    parents.add(cell)
                    fn = lambda parent: fn(parent) + self.weightings[cond] * parent[cell]
                    # note this may result in too much recursion?
            return (fn, parents, cond_weighting)

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

        def init_fn(parent) -> int:
            return 0

        # create a network centered around each node
        cell_networks = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # first create the cpd for this cell
                total_weighting = 0
                current_parents = set()
                current_fn = init_fn
                # determine distribution based on some conditions
                for cond in conditions:
                    if self.weightings[cond] > 0:
                        # add weightings of all cells within 1 distance
                        new_fn, new_parents, new_weighting = add_cond(
                            cond = cond,
                            cells = valid_cells[cond](col, row)
                        )
                        def updated_fn(parent):
                            current_fn(parent) + new_fn(parent)
                        current_fn = updated_fn
                        total_weighting += new_weighting
                        current_parents.union(new_parents)
                cpd = FunctionalCPD(
                    variable=(col, row),
                    #for now, we use a beta distribution with
                    # alpha = (weightings * present values)^2
                    # beta = total weighting
                    fn=lambda parent: distribution.Beta(current_fn(parent)**2, total_weighting),
                    parents=list(current_parents)
                )
                #now we create the network for this cell
                network_edges = self._other_cells(col, row)
                model = FunctionalBayesianNetwork(network_edges)
                model.add_cpds(cpd)
                cell_cpds = self._other_cells(col, row)
                model.add_cpds(cell_cpds)
                cell_networks.update({(col,row): model})


    def fake_board(self, cells: list[tuple[int, int]]) -> list[list[int]]:
        """
        Creates a fake board based on what we know and extra guessed cells
        """
        new_board = [row.copy() for row in self.board]
        cell_value = -1 * ((2*int(self.colour == "w"))-1)
        for cell in cells:
            new_board[cell[1]][cell[0]] = cell_value
        return new_board


    def _empty_cells(self) -> list[tuple[int, int]]:
        """
        Return a list of the empty cells on the board
        """
        return [
            (x,y)
            for x in range(self.num_cols)
            for y in range(self.num_rows)
            if self.board[y][x] == 0
        ]


    def _other_cells(self, col: int, row: int) -> list[tuple[int, int]]:
        """
        Return a list of all cells other than the input one
        """
        return [
            self.cell_cpds[y][x]
            for x in range(self.num_cols)
            for y in range(self.num_rows)
            if x!=col and y!=row
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
        self.env = HexEnv(config={
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
                gamma=0.9,
                lr=0.001,
                kl_coeff=0.3,
                vf_loss_coeff=0.005,
                use_kl_loss=True,
                clip_param=0.2
            )
            .rl_module(
                model_config=DefaultModelConfig(
                    use_lstm=False,
                    fcnet_hiddens=[256, 256],
                    lstm_cell_size=256,
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


    def best_move(self, board: list[list[int]]) -> tuple[int, int]:
        """
        Search for the best move in a given board.
        Returns:
            move: tuple[int, int] - The chosen best move. The top left cell is (0,0), (col, row).
        """
        scores = self.board_scores(board)
        cell_value = (2*int(self.colour == "w"))-1
        best_score = -1000 * cell_value
        best_move = (0,0)
        for row, arr in enumerate(scores):
            for col, val in enumerate(arr):
                if val * cell_value > best_score * cell_value:
                    best_score = val
                    best_move = (col,row)
        return best_move


    def board_scores(self, board: list[list[int]]) -> list[list[float]]:
        """
        Find the expected values of playing in each cell within the board.
        Returns:
            board: list[list[float]] - The scores of every cell on the board.
        """
        #compute the action distribution
        obs_batch = torch.from_numpy({self.colour: board}).unsqueeze(0).float()
        model_outputs = self.rl_module.forward_inference({"obs": obs_batch})
        action_dist = model_outputs["action_dist_inputs"][0].numpy()
        scores = []
        #turn into 2d array
        for row in range(self.num_rows):
            row_scores = []
            for col in range(self.num_cols):
                row_scores.append(action_dist[row*self.num_cols + col])
            scores.append(row_scores)
        return scores


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
        self.abstract_game = None
        self.board = None


    def reset(self, *, seed=None, options=None):
        """
        Initialise the abstract game and set the starting players
        """
        #create abstract game of correct size
        self.abstract_game = dh.AbstractDarkHex(self.num_cols, self.num_rows, turn_check=False)
        self.board = [0] * self.num_cols * self.num_rows
        # return observation dict and infos dict.
        return {"w": np.array(self.board, np.float32)}, {}


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
        col = (action // self.num_rows) + 1
        row = (action % self.num_rows) + 1
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {"w": 0.0, "b": 0.0}
        # Create a terminated-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminated = {"__all__": False}
        #do the move
        #we still use the dark hex abstract game because it works in the same way
        move_result = self.abstract_game.move(col=col, row=row, colour=initial_turn)
        #get rewards, termination, and board updates of the move
        match move_result:
            case "black_win":
                self.board[action] = -1
                rewards["b"] += 10.0
                rewards["w"] -= 10.0
                terminated["__all__"] = True
            case "white_win":
                self.board[action] = 1
                rewards["w"] += 10.0
                rewards["b"] -= 10.0
                terminated["__all__"] = True
            case "full_white":
                # in contrast to dark hex, we never want to play on a full cell
                rewards[initial_turn] -= 100.0
                rewards[util.swap_colour(initial_turn)] += 100.0
            case "full_black":
                rewards[initial_turn] -= 100.0
                rewards[util.swap_colour(initial_turn)] += 100.0
            case "placed":
                # hugely penalize placing a piece in an occupied cell,
                # hence the algorithm shouldn't ever choose to do it
                cell_value = (2*int(initial_turn == "w"))-1
                if self.board[action] == cell_value:
                    #we have played here before
                    rewards[initial_turn] -= 100.0
                    rewards[util.swap_colour(initial_turn)] += 100.0
                else:
                    self.board[action] = cell_value
        turn = self.abstract_game.turn
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return (
            {
                turn: np.array(self.board, np.float32),
            },
            rewards,
            terminated,
            {},
            {}
        )


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
    agent.train(iterations=50)
    logging.info("Agent trained")
    agent.save(
        os.path.join(os.path.dirname(__file__), "trained_agents\\hex_agent_" + str(
            num_cols) + "x" + str(num_rows) + "_" + colour)
    )


if __name__ == "__main__":
    main()
