"""
Module that includes an agent that has an abstracted perfect view of the board,
The imperfect aspect is included using a belief system that the agent never sees.
"""

from agents.agent import Agent
from pgmpy.models import DiscreteBayesianNetwork


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


    def move(self):
        """
        Make a move by the agent. 
        We search for the correct move in multiple belief states,
        and return the one with the highest expected gain
        """


    def update_information(self, col: int, row: int, colour: str):
        """
        Update our belief system with the new information
        """


    def reset(self):
        return super().reset()


    def initialise_network(self):
        """
        Create the base Bayesian Network that contains our beliefs
        """

# For the hex AI, we require some way of returning value of the move,
# ideally the value of multiple moves to add to a large expected value
# accumulation, so we can calculate the true best expected value of a move

# Potentially use Bayesian Networks representing probability of
# them having a piece in a cell rather than Belief States

# We can use the idea that if an opponent has played in one cell, it is fairly likely
# That they have played in another nearby, or in the same line

# Maybe instead of Bayesian Network, we use Markov network,
# And we make the assumption that each game state is memoryless
