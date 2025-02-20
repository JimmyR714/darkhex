"""
Module for the control of the program flow
"""

import logging
import os
import game.display as display
import game.darkhex as darkhex
import agents.agent
import agents.basic_agent
import agents.rl_agent

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
        self.create_agent(
            num_cols=num_cols, num_rows=num_rows, agent=agent, agent_colour=agent_colour
        )
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)

        #check whether the agent needs to move first
        logging.debug("Checking whether agent must move")
        self.agent_move_check(self.game.turn)


    def update_boards(self, row: int, col: int, turn: str, result: str = None) -> None:
        """
        Update each game frame to display the correct board
        """
        if result is None:
            #make the move in the abstract game
            result = self.game.move(row, col, turn)
        #update each game frame appropriately
        #TODO fix bug where pressing your own colour hex tells opponent where it is
        logging.debug("Updating game frames")
        for gf in self.window.game_frames:
            gf.update_board(row, col, result)

        #check whether the agent needs to move
        logging.debug("Checking whether agent must move")
        self.agent_move_check(self.game.turn)


    def agent_move_check(self, turn: str) -> None:
        """
        Checks whether the agent needs to move. 
        If the agent doesn't exist, this will never do anything.
        If the agent does exist but it isn't their turn, this doesn't do anything.
        If the agent does exist and it is their turn, it runs the function for the agent to move.
        It repeatedly runs this function by calling update_boards 
        until they have played a successful move 
        (i.e. if they discover the opponent's hex, they can go again.)
        """
        # allow the agent to move when allowed and until they have played
        if self.agent is not None and self.agent.colour == turn:
            logging.debug("Agent is moving")
            (col, row) = self.agent.move()
            logging.debug("Agent has decided upon (%s, %s)", col, row)
            result = self.game.move(row=row, col=col, colour = turn)
            match result:
                case "full_white":
                    self.agent.update_information(col=col, row=row, colour="w")
                case "full_black":
                    self.agent.update_information(col=col, row=row, colour="b")
                case "placed":
                    self.agent.update_information(col=col, row=row, colour=turn)
            self.update_boards(row, col, turn, result=result)
        else:
            logging.debug("Agent doesn't need to move")


    def create_agent(self, num_cols: int, num_rows: int, agent: str, agent_colour: str) -> None:
        """
        Take the input string describing the agent and create an instance of 
        the corresponding agent.
        """
        logging.debug("Creating a %s agent with colour %s", agent, agent_colour)
        match agent:
            case "General":
                self.agent = agents.agent.Agent(
                    num_cols=num_cols, num_rows=num_rows, colour=agent_colour
                )
            case "Basic":
                self.agent = agents.basic_agent.BasicAgent(
                    num_cols=num_cols, num_rows=num_rows, colour=agent_colour
                )
            case "RL":
                #TODO get path based on settings
                path = os.path.join(os.path.dirname(__file__), "agents\\trained_agents\\rl_agent")
                self.agent = agents.rl_agent.RLAgent.from_file(
                    path=path
                )
                self.agent.reset()
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
