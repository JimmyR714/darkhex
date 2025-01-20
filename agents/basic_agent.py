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
    def __init__(self, num_cols: int, num_rows: int, colour: str):
        """Create a basic agent"""
        super().__init__(num_cols, num_rows, colour)
        self.belief_state = BeliefState(self.colour)


    def move(self):
        """
        Make a move by the agent.
        Moves are determined by the one that results in the belief state with the 
        highest probability of our success.
        """
        return self.belief_state.optimal_move()

    def update_information(self, col, row, colour):
        return self.belief_state.update_information(col, row, colour)


class BeliefState():
    """
    Class that represents a belief state for the agent.
    Stores a minimal amount of information, but has some useful functions
    """
    def __init__(self, agent_colour: str):
        self.agent_colour = agent_colour


    def optimal_move(self) -> tuple[int, int]:
        """
        Returns the optimal move in the current belief state
        Uses alpha-beta pruned minimax search
        """
        def min_value(state: BeliefState, alpha: float, beta: float) -> float:
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
                # try this action, update min value if necessary
                current_min_value = min(
                    current_min_value,
                    max_value(state.result(action), alpha, beta)
                )
                # attempt pruning
                if current_min_value <= alpha:
                    return current_min_value
                #update pruning value
                beta = min(beta, current_min_value)

            return current_min_value


        def max_value(state: BeliefState, alpha: float, beta: float) -> float:
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
                # try this action, update max value if necessary
                current_max_value = max(
                    current_max_value,
                    min_value(state.result(action), alpha, beta)
                )
                # attempt pruning
                if current_max_value >= beta:
                    return current_max_value
                # update pruning value
                alpha = max(alpha, current_max_value)

            return current_max_value

        best_action = None
        best_utility = None
        for action in self.actions():
            utility_value = min_value(self.result(action), -1000000000.0, 1000000000.0)
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
        return self.agent_colour != colour


    def actions(self) -> list[tuple[int, int]]:
        """
        Returns a list of the moves possible in the current belief state
        Returns the action that we think is most likely to benefit us first, 
        and least likely to benefit us last.
        """


    def result(self, action: tuple[int, int]) -> Self:
        """
        Returns the belief state reached by applying action in state
        """


    def terminal_test(self) -> bool:
        """
        Determine whether we should terminate the search
        """


    def utility(self) -> float:
        """
        The utility of the belief state
        """
