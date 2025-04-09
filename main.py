"""
Main module for the control of the program flow.

To add an agent to the system:
- Add a case to the create agent function for the new name of the agent
- Add the agent name to the agent_options list in display.MainMenuFrame.update_game_menu
- Add any agent settings to display.MainMenuFrame.update_agent_menu
"""

import logging
import os
import agents.abstract_agent
import game.util as util
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
        self.agent2: agents.agent.Agent = None


    def make_window(self) -> None:
        """
        Create the display window for the program
        """
        self.window = display.DisplayWindow(self)
        self.window.mainloop()


    def new_game(self, num_cols: int, num_rows: int) -> None:
        """
        Create a new game of dark hex
        """
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)


    def new_pva_game(self, agent_settings: dict):
        """
        Create a player vs agent game
        Prerequisite: a new game must have been created
        """
        self.create_agent(agent_settings=agent_settings)
        #check whether the agent needs to move first
        logging.debug("Checking whether agent must move")
        self.agent_move_check(self.game.turn)


    def new_ava_game(self, agent_1_settings: dict, agent_2_settings: dict, iterations: int):
        """
        Create an agent vs agent game
        Prerequisite: a new game must have been created
        """
        self.create_agent(agent_settings=agent_1_settings)
        self.create_agent(agent_settings=agent_2_settings, second=True)
        self.simulate_ava_game(iterations)


    def simulate_ava_game(self, iterations: int = 1):
        """
        Simulates a certain number of iterations of agent vs agent games
        """
        agent_1_wins = 0
        agent_2_wins = 0
        agent_1_colour = self.agent.colour
        #play many games
        for game_num in range(iterations):
            logging.debug("Starting agent vs agent game %s", game_num+1)
            move_result = None
            while move_result not in ["white_win", "black_win"]:
                #play moves until one wins
                initial_turn = self.game.turn
                #check whose turn it is
                if initial_turn == agent_1_colour:
                    #agent 1's turn
                    this_agent = self.agent
                else:
                    #agent 2's turn
                    this_agent = self.agent2
                #let the correct agent move
                (col, row) = this_agent.move()
                #check the results of the move
                move_result = self.game.move(row=row, col=col, colour = initial_turn)
                #update the information of the correct agent
                match move_result:
                    case "full_white":
                        this_agent.update_information(col=col, row=row, colour="w")
                    case "full_black":
                        this_agent.update_information(col=col, row=row, colour="b")
                    case "placed":
                        this_agent.update_information(col=col, row=row, colour=initial_turn)
            if (agent_1_colour == "w") == (move_result == "white_win"):
                #agent 1 has won
                agent_1_wins += 1
                logging.debug("Agent 1 has won")
            else:
                agent_2_wins += 1
                logging.debug("Agent 2 has won")
            #reset game
            self.game.reset_board()
            #reset agents
            self.agent.reset()
            self.agent2.reset()
        logging.debug("Agent 1 won %s games; Agent 2 won %s games", agent_1_wins, agent_2_wins)


    def update_boards(self, row: int, col: int, turn: str, result: str = None) -> None:
        """
        Update each game frame to display the correct board
        """
        if result is None:
            #make the move in the abstract game
            result = self.game.move(row, col, turn)
        #update each game frame appropriately
        if (result != "full_white" or turn != "w") and (result != "full_black" or turn != "b"):
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


    def create_agent(self, agent_settings: dict, second= False) -> None:
        """
        Take the input string describing the agent and create an instance of 
        the corresponding agent.
        If second is true, it creates a second agent
        Prerequisite: a game must have been created prior to this
        """
        num_cols = self.game.cols
        num_rows = self.game.rows
        agent = agent_settings["type"]
        logging.debug("Creating a %s agent with settings %s", agent, agent_settings)
        match agent:
            case "General":
                new_agent = agents.agent.Agent(
                    num_cols=num_cols, num_rows=num_rows, settings=agent_settings
                )
            case "Basic":
                new_agent = agents.basic_agent.BasicAgent(
                    num_cols=num_cols, num_rows=num_rows, settings=agent_settings
                )
            case "RL":
                new_agent = agents.rl_agent.RLAgent.from_file(
                    path=util.select_rl_agent(
                        cols=num_cols,
                        rows=num_rows,
                        colour=agent_settings["colour"],
                        current_path=os.path.dirname(__file__)
                    )
                )
                new_agent.reset()
            case "Abstract":
                new_agent = agents.abstract_agent.AbstractAgent(
                    num_cols=num_cols,
                    num_rows=num_rows,
                    settings=agent_settings
                )
            case _:
                raise ValueError(f"Agent type \"{agent}\" does not exist.")
        #check which number agent this is
        if second:
            self.agent2 = new_agent
        else:
            self.agent = new_agent


def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    controller = Controller()
    controller.make_window()


if __name__ == "__main__":
    main()
