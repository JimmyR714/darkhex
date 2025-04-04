"""
Module that includes an agent that has an abstracted perfect view of the board,
The imperfect aspect is included using a belief system that the agent never sees.
"""

from agents.agent import Agent
from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD
import pyro.distributions as dist


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
        self.cell_networks = self.initialise_networks()
        self.weightings = settings["weightings"]


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
        self.cell_networks = self.initialise_networks()
        return super().reset()


    def initialise_networks(self) -> list[list[FunctionalBayesianNetwork]]:
        """
        Create the Bayesian Networks that contains our beliefs
        """
        # create a network centered around each node
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                total_weighting = 0
                # first, create our model using only cpds based on distance from us
                if self.weightings["width_1"] > 0:
                    # add weightings of all cells within 1 distance
                    width_1_cpd = FunctionalCPD(
                        variable=(col, row),
                        fn=self.weightings["width_1"], # add function here
                        parents=[] #add parents here
                    )
                if self.weightings["width_2"] > 0:
                    # add weightings of all cells within 2 distance
                    width_2_cpd = FunctionalCPD()
                

# For the hex AI, we require some way of returning value of the move,
# ideally the value of multiple moves to add to a large expected value
# accumulation, so we can calculate the true best expected value of a move

# Potentially use Bayesian Networks representing probability of
# them having a piece in a cell rather than Belief States

# We can use the idea that if an opponent has played in one cell, it is fairly likely
# That they have played in another nearby, or in the same line

# Maybe instead of Bayesian Network, we use Markov network,
# And we make the assumption that each game state is memoryless
