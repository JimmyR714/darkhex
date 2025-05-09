"""
Main module for the control of the program flow.

To add an agent to the system:
- Add a case to the create agent function for the new name of the agent
- Add the agent name to the agent_options list in display.MainMenuFrame.update_game_menu
- Add any agent settings to display.MainMenuFrame.update_agent_menu
"""

import logging
import os
import game.util as util
import game.darkhex as darkhex
from display.display_window import DisplayWindow
import agents.agent
import agents.basic_agent
import agents.rl_agent
import agents.abstract_agent


class Controller:
    """
    Handles the high-level control over the abstract dark hex
    and the display window, as well as (in future), AI
    """
    def __init__(self):
        """
        Set constants necessary (maybe from a config file)
        """
        self.window: DisplayWindow = None
        self.game: darkhex.AbstractDarkHex = None
        self.agent: agents.agent.Agent = None
        self.agent2: agents.agent.Agent = None
        self.learning = False
        self.referee_board = []


    def make_window(self) -> None:
        """
        Create the display window for the program
        """
        self.window = DisplayWindow(self)
        self.window.mainloop()


    def new_game(self, num_cols: int, num_rows: int) -> None:
        """
        Create a new game of dark hex
        """
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)


    def new_pva_game(self, agent_settings: dict):
        """
        Create a player vs agent game.
        
        Prerequisite: a new game must have been created
        """
        if self.agent is not None:
            #reset the agent
            #TODO we cannot change agent types while running now
            self.agent.reset(self.referee_board)
        else:
            self.create_agent(agent_settings=agent_settings)
        #check whether the agent needs to move first
        logging.debug("Checking whether agent must move")
        self.agent_move_check(self.game.turn)


    def new_ava_game(self, agent_1_settings: dict, agent_2_settings: dict, iterations: int) -> int:
        """
        Create an agent vs agent game
        Prerequisite: a new game must have been created
        Returns:
            agent_1_wins: int - How many games agent 1 won
        """
        self.create_agent(agent_settings=agent_1_settings)
        self.create_agent(agent_settings=agent_2_settings, second=True)
        return self.simulate_ava_game(iterations)


    def simulate_ava_game(self, iterations: int = 1) -> int:
        """
        Simulates a certain number of iterations of agent vs agent games
        Returns:
            agent_1_wins: int - How many games agent 1 won
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
            #reset agents, send referee's board in case they can learn
            self.agent.reset(self.game.board)
            self.agent2.reset(self.game.board)
            #reset game
            self.game.reset_board()
        logging.info("Agent 1 won %s games; Agent 2 won %s games", agent_1_wins, agent_2_wins)
        return agent_1_wins


    def update_boards(self, row: int, col: int, turn: str, result: str = None) -> None:
        """
        Update each game frame to display the correct board
        """
        #update the referee's board
        self.referee_board = self.game.board
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


    def select_rl_agent(self, cols: int, rows: int, colour: str,
                    current_path: str, agent_type: str = "rl_agent") -> str:
        """
        Select the correct agent path based on game settings
        Returns the full correct path.
        """
        path_1 = "agents\\trained_agents\\" + agent_type
        agent_path = path_1 + "_" + str(cols) + "x" + str(rows) + "_" + colour
        path = os.path.join(current_path, agent_path)
        if not os.path.exists(path):
            #train a new agent
            match agent_type:
                case "rl_agent":
                    agent = agents.rl_agent.RLAgent.to_train(
                        num_cols=cols, num_rows=rows, colour=colour
                    )
                case "hex_agent":
                    agent = agents.abstract_agent.HexAgent.to_train(
                        num_cols=cols, num_rows=rows, colour=colour
                    )
            logging.info("Training Agent")
            agent.train(iterations=100)
            logging.info("Agent trained")
            logging.debug(agent.save(
                os.path.join(os.path.dirname(__file__), path_1 + "_" + str(
                    cols) + "x" + str(rows) + "_" + colour)
            ))
        else:
            logging.info("Fetching agent from %s", path)
        return path


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
                    path=self.select_rl_agent(
                        cols=num_cols,
                        rows=num_rows,
                        colour=agent_settings["colour"],
                        current_path=os.path.dirname(__file__)
                    )
                )
                new_agent.reset(self.game.board)
            case "Bayesian":
                #change settings from intvars
                if not isinstance(agent_settings["width_1"], int):
                    agent_settings.update(
                        {
                            "width_1": int(agent_settings["width_1"].get()),
                            "width_2_vc": int(agent_settings["width_2_vc"].get()),
                            "width_2_semi_vc": int(agent_settings["width_2_semi_vc"].get()),
                            "learning": agent_settings["learning"].get(),
                            "fake_boards": int(agent_settings["fake_boards"].get())
                        }
                    )
                new_agent = agents.abstract_agent.AbstractAgent(
                    num_cols=num_cols,
                    num_rows=num_rows,
                    hex_path=self.select_rl_agent(
                        cols=num_cols,
                        rows=num_rows,
                        colour=agent_settings["colour"],
                        current_path=os.path.dirname(__file__),
                        agent_type="hex_agent"
                    ),
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
