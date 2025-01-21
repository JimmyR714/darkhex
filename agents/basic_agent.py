"""
Module for a very basic agent that
works solely off of the rules of dark hex
"""
from typing import Self
import agents.agent

class BasicAgent(agents.agent.Agent):
    """
    Basic agent that uses the rules of dark hex, but no advanced techniques.
    """
    MAX_DEPTH = 3
    def __init__(self, num_cols: int, num_rows: int, colour: str):
        """Create a basic agent"""
        super().__init__(num_cols, num_rows, colour)
        self.belief_state = BeliefState(
            board = self.board,
            agent_colour=self.colour,
            max_depth=self.MAX_DEPTH
        )


    def move(self):
        """
        Make a move by the agent.
        Moves are determined by the one that results in the belief state with the 
        highest probability of our success.
        """
        #TODO must replace old belief state with new one?
        return self.belief_state.optimal_move()

    def update_information(self, col, row, colour):
        return self.belief_state.update_information(col, row, colour)


class BeliefState():
    """
    Class that represents a belief state for the agent.
    Stores a minimal amount of information, but has some useful functions
    """
    def __init__(self, board: list[list[str]], agent_colour: str, max_depth: int):
        self.agent_colour = agent_colour
        self.board = board
        self.num_rows = len(board)
        self.num_cols = len(board[0])
        self.max_depth = max_depth


    def optimal_move(self) -> tuple[int, int]:
        """
        Returns the optimal move in the current belief state
        Uses alpha-beta pruned minimax search
        """
        def min_value(state: BeliefState, alpha: float, beta: float, white_turn: bool) -> float:
            """
            Returns the minimum expected utility value in state
            """
            # check for termination
            if state.terminal_test():
                return state.utility()
            # search for min value
            current_min_value = 1000000000.0
            # check each possible action
            for action in state.actions():
                for result in self.results(white_turn, action):
                    # try this action, update min value if necessary
                    current_min_value = min(
                        current_min_value,
                        max_value(result, alpha, beta, white_turn=not white_turn)
                    )
                    # attempt pruning
                    if current_min_value <= alpha:
                        return current_min_value
                    #update pruning value
                    beta = min(beta, current_min_value)

            return current_min_value


        def max_value(state: BeliefState, alpha: float, beta: float, white_turn: bool) -> float:
            """
            Returns the maximum expected utility value in state
            """
            # check for termination
            if state.terminal_test():
                return state.utility()
            # search for max value
            current_max_value = -1000000000.0
            # check each possible action
            for action in state.actions():
                for result in self.results(white_turn, action):
                    # try this action, update max value if necessary
                    current_max_value = max(
                        current_max_value,
                        min_value(result, alpha, beta, white_turn=not white_turn)
                    )
                    # attempt pruning
                    if current_max_value >= beta:
                        return current_max_value
                    # update pruning value
                    alpha = max(alpha, current_max_value)

            return current_max_value

        white_turn = self.agent_colour=="w"
        best_action = None
        best_utility = None
        for action in self.actions():
            for result in self.results(white_turn, action):
                utility_value = min_value(
                    result, -1000000000.0, 1000000000.0, white_turn
                )
                if best_utility is None or utility_value > best_utility:
                    best_action = action
                    best_utility = utility_value

        return best_action


    def update_information(self, col: int, row: int, colour: str) -> bool:
        """
        Recieve some updated information about the board.
        The agent with this belief state will call this after we make a move.
        Necessary to find out whether our move was successful.
        
        Parameters:
            col: int: The chosen column to play in.
                col = 1 is the left column.
            row: int: The chosen row to play in.
                row = 1 is the top row.
            colour: str: The colour of the new cell. "w" for white and "b" for black.
            
        Returns:
            True: When another move must be made.
            False: When no other move must be made.
        """
        #TODO maintain connected components like in abstract dark hex implementation
        return self.agent_colour != colour


    def actions(self) -> list[tuple[int, int]]:
        """
        Returns a list of the moves possible in the current belief state
        Returns the action that we think is most likely to benefit us first, 
        and least likely to benefit us last.
        """
        #TODO add move ordering based on proximity to connected components
        #naive implementation with no ordering
        possible_actions = []
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                if self.board[row][col] == "e":
                    possible_actions.append((col, row))

        return possible_actions


    def results(self, white_turn: bool, action: tuple[int, int]) -> list[Self]:
        """
        Returns the belief states reached by applying action in state
        """
        #return both possible boards to result from this position
        #TODO both boards will not always be possible
        new_board_white = self.board.copy()
        new_board_black = self.board.copy()
        new_board_white[action[1]][action[0]] = "w"
        new_board_black[action[1]][action[0]] = "b"
        return [
            BeliefState(new_board_white,
                        agent_colour=self.agent_colour,
                        max_depth=self.max_depth
            ),
            BeliefState(new_board_black,
                        agent_colour=self.agent_colour,
                        max_depth=self.max_depth
            )
        ]


    def terminal_test(self) -> bool:
        """
        Determine whether we should terminate the search
        """
        #TODO add win check
        if self.max_depth == 0:
            return True


    def utility(self) -> float:
        """
        The utility of the belief state
        """
        #TODO use connected components to improve utility function
