"""
Module for agents that use reinforcement learning
"""
from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
import numpy as np
import agents.agent
import game.darkhex as dh

class RLAgent(agents.agent.Agent):
    """
    Agent that uses Rl techniques to choose moves
    """
    USE_LTSM = True
    def __init__(self, num_cols, num_rows, colour):
        super().__init__(num_cols, num_rows, colour)

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
                    use_lstm=self.USE_LTSM,
                    # Use a simpler FCNet when we also have an LSTM.
                    fcnet_hiddens=[32] if self.USE_LTSM else [256, 256],
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

        self.algo = config.build_algo()


    def train(self, iterations = 5):
        """
        Train the algorithm and print the results
        """
        # Train it for 5 iterations
        for _ in range(iterations):
            pprint(self.algo.train())

        # evaluate the algorithm
        pprint(self.algo.evaluate())


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
        action = action_dict[self.abstract_game.turn]
        col = action / self.num_rows
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
    agent = RLAgent(num_cols=num_cols, num_rows=num_rows, colour="w")
    agent.train(iterations=1)

if __name__ == "__main__":
    main()
