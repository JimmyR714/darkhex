"""
Module for the control of the program flow
"""

import logging
import game.display as display
import game.darkhex as darkhex
from agents.agent import *

class Controller:
    """
    Handles the high-level control over the abstract dark hex
    and the display window, as well as (in future), AI
    """
    def __init__(self):
        """
        Set constants necessary (maybe from a config file)
        """
        self.window: display.DisplayWindow = None 
        self.game: darkhex.AbstractDarkHex = None
        self.agent: Agent = None


    def make_window(self):
        """
        Create the display window for the program
        """
        self.window = display.DisplayWindow(self)
        self.window.mainloop()


    def new_game(self, num_cols: int, num_rows: int, agent: str):
        """
        Create a new game of dark hex
        """
        self.agent = agent
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)


    def update_boards(self, row: int, col: int, turn: str):
        """
        Update each game frame to display the correct board
        """
        #make the move in the abstract game
        result = self.game.move(row, col, turn)
        #update each game frame appropriately
        for gf in self.window.game_frames:
            gf.update_board(row, col, result)

        #check whether the ai needs to move
        self.agent_move_check(self.game.turn)


    def agent_move_check(self, turn: str):
        """
        Checks whether the agent needs to move. 
        If the agent doesn't exist, this will never do anything.
        If the agent does exist but it isn't their turn, this doesn't do anything.
        If the agent does exist and it is their turn, it runs the function for the agent to move.
        It repeatedly runs this function until they have played a successful move 
        (i.e. if they discover the opponent's hex, they can go again.)
        """
        # check if the agent should move
        if self.agent is not None and self.agent.colour == turn:
            cell = self.agent.move()

def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    controller = Controller()
    controller.make_window()

if __name__ == "__main__":
    main()
