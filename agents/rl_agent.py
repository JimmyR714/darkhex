"""
Module for agents that use reinforcement learning
"""
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import agents.agent

class RLAgent(agents.agent.Agent):
    """
    Agent that uses Rl techniques to choose moves
    """
    def __init__(self, num_cols, num_rows, colour):
        super().__init__(num_cols, num_rows, colour)

        #define rl algorithm
        config = (
            PPOConfig()
            .environment("Pendulum-v1")
        )
        config.env_runners(num_env_runners=2)
        config.training(
            lr=0.0002,
            train_batch_size_per_learner=2000,
            num_epochs=10,
        )

        self.algo = config.build_algo()
        

class DarkHexEnv(MultiAgentEnv):
    """
    Environment to be passed into PPO algorithm
    """
    def __init__(self, config=None):
        super().__init__()
        self.possible_agents = ["white", "black"]

    def reset(self, *, seed=None, options=None):
        # return observation dict and infos dict.
        return #{"white": [obs of agent_1], "black": [obs of agent_2]}, {}


    def step(self, action_dict):
        # return observation dict, rewards dict, termination/truncation dicts, and infos dict
        return #{"white": [obs of agent_1]}, {...}, ...
