"""
Module for the control of the program flow
"""

import logging
import game.display as display
import game.darkhex as darkhex
import agents.agent
import agents.basic_agent

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
        self.agent: agents.agent.Agent = None


    def make_window(self) -> None:
        """
        Create the display window for the program
        """
        self.window = display.DisplayWindow(self)
        self.window.mainloop()


    def new_game(self, num_cols: int, num_rows: int, agent: str, agent_colour: str) -> None:
        """
        Create a new game of dark hex
        """
        self.create_agent(num_cols=num_cols, num_rows=num_rows, agent=agent, agent_colour=agent_colour)
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)


    def update_boards(self, row: int, col: int, turn: str) -> None:
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


    def agent_move_check(self, turn: str) -> None:
        """
        Checks whether the agent needs to move. 
        If the agent doesn't exist, this will never do anything.
        If the agent does exist but it isn't their turn, this doesn't do anything.
        If the agent does exist and it is their turn, it runs the function for the agent to move.
        It repeatedly runs this function until they have played a successful move 
        (i.e. if they discover the opponent's hex, they can go again.)
        """
        # allow the agent to move when allowed and until they have played
        while self.agent is not None and self.agent.colour == turn:
            (col, row) = self.agent.move()
            result = self.game.move(row=row, col=col, colour = turn)
            match result:
                case "full_white":
                    self.agent.update_information(col=col, row=row, colour="w")
                case "full_black":
                    self.agent.update_information(col=col, row=row, colour="b")
                case "placed":
                    self.agent.update_information(col=col, row=row, colour=turn)


    def create_agent(self, num_cols: int, num_rows: int, agent: str, agent_colour: str) -> None:
        """
        Take the input string describing the agent and create an instance of 
        the corresponding agent.
        """
        match agent:
            case "general":
                self.agent = agents.agent.Agent(
                    num_cols=num_cols, num_rows=num_rows, colour=agent_colour)
            case "basic":
                self.agent = agents.basic_agent.Basic_Agent(
                    num_cols=num_cols, num_rows=num_rows, colour=agent_colour)
            case None: # no agent, a 2 player game
                self.agent = None
            case _:
                raise ValueError(f"Agent type \"{agent}\" does not exist.")

def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    controller = Controller()
    controller.make_window()

if __name__ == "__main__":
    main()
