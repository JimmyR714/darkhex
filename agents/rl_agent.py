"""
Module for agents that use reinforcement learning
"""
from pprint import pprint
import os
import logging
import json
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module import RLModule
import torch
import gymnasium as gym
import numpy as np
import agents.agent
import game.darkhex as dh

class RLAgent(agents.agent.Agent):
    """
    Agent that uses Rl techniques to choose moves
    """
    def __init__(self, num_cols: int, num_rows: int, colour: str,
                 rl_module: RLModule, algo: Algorithm = None):
        super().__init__(num_cols, num_rows, colour)
        self.rl_module = rl_module
        self.algo = algo
        self.env = DarkHexEnv(config={
                    "num_cols": num_cols,
                    "num_rows": num_rows
                })
        self.obs = None
        self.info = None


    @classmethod
    def from_file(cls, path: str):
        """
        Retrieve an agent from a file path.
        The number of cols and rows must also be stored in that path
        """
        with open(os.path.join(path, "settings.json"), "r", encoding="utf-8") as f:
            settings = json.load(f)
        rl_module = RLModule.from_checkpoint(
            path
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )
        return cls(
            num_cols=settings["num_cols"],
            num_rows=settings["num_rows"],
            colour=settings["colour"],
            rl_module=rl_module
        )


    @classmethod
    def to_train(cls, num_cols: int, num_rows: int, colour: str):
        """
        Create a fresh, untrained agent
        """
        use_ltsm = True
        #define rl algorithm
        config = (
            PPOConfig()
            .env_runners(
                env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
            )
            .environment(
                DarkHexEnv,
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
                vf_loss_coeff=0.005,
            )
            .rl_module(
                model_config=DefaultModelConfig(
                    use_lstm=use_ltsm,
                    # Use a simpler FCNet when we also have an LSTM.
                    fcnet_hiddens=[32] if use_ltsm else [256, 256],
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
            colour=colour,
            rl_module=None,
            algo = config.build_algo(),
        )


    def reset(self):
        """
        Reset the environment that the agent uses.
        Must be used at the start of a game.
        """
        self.obs, self.info = self.env.reset()


    def move(self):
        #compute the next action
        obs_batch = torch.from_numpy(self.obs).unsqueeze(0)
        model_outputs = self.rl_module.forward_inference({"obs": obs_batch})

        #extract action distribution
        action_dist = model_outputs["action_dist_inputs"][0].numpy()

        #get most likely action
        best_action = np.argmax(action_dist)

        #play the action
        #TODO do I need the other variables?
        self.obs, reward, terminated, truncated, info = self.env.step(best_action)

        #return move for abstract game
        return (best_action // self.num_rows, best_action % self.num_rows)


    def train(self, iterations = 5):
        """
        Train the algorithm and print the results
        """
        # Train it for 5 iterations
        for _ in range(iterations):
            pprint(self.algo.train())
        #save to our rl module
        self.rl_module = self.algo.get_module(self.colour)


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


class DarkHexEnv(MultiAgentEnv):
    """
    Environment to be passed into PPO algorithm
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
        self.abstract_game = dh.AbstractDarkHex(self.num_cols, self.num_rows)
        self.white_board = self.black_board = [0] * self.num_cols * self.num_rows
        # return observation dict and infos dict.
        return {"w": np.array(self.white_board, np.float32)}, {}


    def step(self, action_dict):
        """
        Do one step in this episode
        """
        #action is (col * num_rows) + row
        action : int = action_dict[self.abstract_game.turn]
        col = action // self.num_rows
        row = action % self.num_rows
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {"w": 0.0, "b": 0.0}
        # Create a terminated-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminated = {"__all__": False}
        #do the move
        initial_turn = self.abstract_game.turn
        move_result = self.abstract_game.move(col=col, row=row, colour=initial_turn)
        #get rewards, termination, and board updates of the move
        match move_result:
            case "black_win":
                self.black_board[action] = -1
                rewards["b"] += 10.0
                rewards["w"] -= 10.0
                terminated["__all__"] = True
            case "white_win":
                self.white_board[action] = 1
                rewards["w"] += 10.0
                rewards["b"] -= 10.0
                terminated["__all__"] = True
            case "full_white":
                self.black_board[action] = 1
            case "full_black":
                self.white_board[action] = -1
            case "placed":
                self.get_board(initial_turn)[action] = (2*int(initial_turn == "w"))-1
        turn = self.abstract_game.turn
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return (
            {turn:
                np.array(self.get_board(turn), np.float32)},
            rewards,
            terminated,
            {},
            {}
        )


    def get_board(self, colour: str) -> list[int]:
        """
        Returns the board for the given colour
        """
        if colour == "w":
            return self.white_board
        else:
            return self.black_board


#run to train the agent
def main():
    """
    Train the agent on a certain board size
    """
    num_cols = 3
    num_rows = 3
    agent = RLAgent.to_train(num_cols=num_cols, num_rows=num_rows, colour="w")
    logging.info("Training Agent")
    agent.train()
    logging.info("Agent trained")
    logging.debug(agent.save(os.path.join(os.path.dirname(__file__), "rl_agent_checkpoint")))

if __name__ == "__main__":
    main()
