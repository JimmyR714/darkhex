"""
Module for a very basic agent that
works solely off of the rules of dark hex
"""
import agents.agent

class BasicAgent(agents.agent.Agent):
    """
    Basic agent that uses the rules of dark hex, but no advanced techniques.
    """
    def __init__(self, num_cols: int, num_rows: int, colour: str):
        """Create a basic agent"""
        super().__init__(num_cols, num_rows, colour)
        #other stuff


    def move(self):
        """
        Make a move by the agent.
        Moves are determined by...
        """

        chosen_col = 1
        chosen_row = 1

        return (chosen_col, chosen_row)
