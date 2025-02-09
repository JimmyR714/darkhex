"""
Module for agents that use reinforcement learning
"""
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
import agents.agent
import game.darkhex as dh

class RLAgent(agents.agent.Agent):
    """
    Agent that uses Rl techniques to choose moves
    """
    def __init__(self, num_cols, num_rows, colour):
        super().__init__(num_cols, num_rows, colour)

        #define rl algorithm
        config = (
            PPOConfig()
            .environment(DarkHexEnv)
            .env_runners(num_env_runners=2)
            .training(
                lr=0.0002,
                train_batch_size_per_learner=2000,
                num_epochs=10,
            )
        )

        self.algo = config.build_algo()


class DarkHexEnv(MultiAgentEnv):
    """
    Environment to be passed into PPO algorithm
    """
    def __init__(self, num_cols: int, num_rows: int):
        super().__init__()
        #define the agents in the game
        self.agents = self.possible_agents = ["white", "black"]

        #each agent observes a space of size num_cols x num_rows
        #each cell can range from -1.0 to 1.0, with -1.0 being black and 1.0 being white
        self.observation_spaces = {
            "w": gym.spaces.Box(-1.0, 1.0, (num_cols, num_rows,), np.float32),
            "b": gym.spaces.Box(-1.0, 1.0, (num_cols, num_rows,), np.float32)
        }

        #each agent can place in any of the cells
        self.action_spaces = {
            "w": gym.spaces.Discrete(num_cols * num_rows),
            "b": gym.spaces.Discrete(num_cols * num_rows),
        }

        self.num_cols = num_cols
        self.num_rows = num_rows
        self.abstract_game = None
        self.current_player = None


    def reset(self, *, seed=None, options=None):
        """
        Initialise the abstract game and set the starting players
        """
        #create abstract game of correct size
        self.abstract_game = dh.AbstractDarkHex(self.num_cols, self.num_rows)
        # return observation dict and infos dict.
        return {"w": np.array(self.abstract_game.white_board)}, {}


    def step(self, action_dict):
        """
        Do one step in this episode
        """
        #action is (col, row) tuple
        action = action_dict[self.abstract_game.turn]
        # Create a rewards-dict (containing the rewards of the agent that just acted).
        rewards = {"w": 0.0, "b": 0.0}
        # Create a terminated-dict with the special `__all__` agent ID, indicating that
        # if True, the episode ends for all agents.
        terminated = {"__all__": False}
        #do the move
        move_result = self.abstract_game.move(action[1], action[0], self.abstract_game.turn)
        match move_result:
            case "black_win":
                rewards["b"] += 100.0
                rewards["w"] -= 100.0
                terminated["__all__"] = True
            case "white_win":
                rewards["w"] += 100.0
                rewards["b"] -= 100.0
                terminated["__all__"] = True
            case "full_white":
                rewards["b"] += 1.0
            case "full_black":
                rewards["w"] += 1.0
            case "placed":
                pass
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return (
            {self.abstract_game.turn:
                np.array(self.abstract_game.get_board(self.abstract_game.turn), np.float32)},
            rewards,
            terminated,
            {},
            {}
        )
