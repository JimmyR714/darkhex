"""
Module for a very basic agent that
works solely off of the rules of dark hex
"""
import agent

class Basic_Agent(agent.Agent):
    def __init__(self, num_cols: int, num_rows: int, colour: str):
        """Create a basic agent"""
        super().__init__(num_cols, num_rows, colour)
        #other stuff


    #def move(self):
        """
        Make a move by the agent.
        Moves are determined by...
        """
