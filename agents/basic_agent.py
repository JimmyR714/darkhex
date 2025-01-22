"""
Module for a very basic agent that
works solely off of the rules of dark hex
"""
from typing import Self
from copy import deepcopy
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
            initial_beliefs=[
                {
                    "prob": 1,
                    "board": [["e"] * num_cols] * num_rows
                }
            ],
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
    Maintains a list of beliefs about our current state.
    Each belief is a full amount of information about the board, hence applying any
    action to a belief is deterministic based on that belief being true.
    """
    def __init__(self, initial_beliefs: list[dict], agent_colour: str, max_depth: int):
        self.agent_colour = agent_colour
        self.num_rows = len(initial_beliefs[0]["board"])
        self.num_cols = len(initial_beliefs[0]["board"][0])
        self.max_depth = max_depth
        """
        we maintain a list of beliefs about our current state
        each belief contains the probability of the belief being true
        and the board represented by the belief
        'e' means we believe the cell is empty
        'bu' means we believe the cell is black and unseen by white
        'bs' means we believe the cell is black and seen by white
        'wu' means we believe the cell is white and unseen by black
        'ws' means we believe the cell is white and seen by black
        """
        self.beliefs = initial_beliefs
        #we always believe the same spaces may be empty (or containing an unseen opponent)
        #irrespective of the list of beliefs. Hence we maintain this:
        placable_cells : list[tuple[int, int]] = []
        for col in range(num_cols):
            for row in range(num_rows):
                placable_cells.append((col, row))
        self.placable_cells = placable_cells


    def opponent_move(self) -> None:
        """
        Performs the required updates to the beliefs after an opponent has performed a move.
        We take all of our current beliefs and include the possibilities for the opponent moving.
        Must be called every time the opponent has moved to allow us to change our beliefs.
        """
        #TODO each probability of move is assumed equal in a given belief. we should take account of
        #TODO the goals of the opponent, i.e. they should think like us in theory
        new_beliefs = []
        for belief in self.beliefs:
            # if a cell is empty, they might have placed their piece there.
            # if a cell of our colour is unseen, they might have found it
            for row in belief["board"]:
                for cell in row:
                    #TODO opponent belief updates just below
                    if cell == "e":
                        #they might have placed their piece here
                        pass
                    elif cell == "wu" and self.agent_colour == "w":
                        #they might have found our piece
                        pass
                    elif cell == "bu" and self.agent_colour == "b":
                        #they might have found our piece
                        pass
        self.beliefs = new_beliefs


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
            for action in state.placable_cells:
                # try this action, update min value if necessary
                current_min_value = min(
                    current_min_value,
                    max_value(
                        self.result(white_turn, action),
                        alpha,
                        beta,
                        white_turn=not white_turn
                    )
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
            for action in state.placable_cells:
                # try this action, update max value if necessary
                current_max_value = max(
                    current_max_value,
                    min_value(
                        self.result(white_turn, action),
                        alpha,
                        beta,
                        white_turn=not white_turn
                    )
                )
                # attempt pruning
                if current_max_value >= beta:
                    return current_max_value
                # update pruning value
                alpha = max(alpha, current_max_value)

            return current_max_value

        # start the minimax search
        white_turn = self.agent_colour=="w"
        best_action = None
        best_utility = None
        for action in self.placable_cells:
            utility_value = min_value(
                self.result(white_turn, action), -1000000000.0, 1000000000.0, white_turn
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
        #update our beliefs based on this new information
        new_beliefs = []
        for belief in self.beliefs:
            if colour == self.agent_colour:
                #we successfully placed our piece, and it is unseen
                new_belief = belief["board"]
                new_belief[row][col] = self.agent_colour + "u"
                new_beliefs.append(new_belief)
            elif colour != self.agent_colour and belief["board"][row][col] == colour + "u":
                #we discovered one of their pieces,
                # hence remove all beliefs where this wasn't unseen
                new_belief = belief["board"]
                new_belief[row][col] = colour + "s"
                new_beliefs.append(new_belief)
        self.beliefs = new_beliefs

        #we can no longer place a piece in this cell
        self.placable_cells.remove((col, row))

        #TODO maintain connected components like in abstract dark hex implementation
        return self.agent_colour != colour


    def result(self, white_turn: bool, action: tuple[int, int]) -> Self:
        """
        Returns the belief state reached by applying action to each of our beliefs
        """
        if white_turn:
            colour = "w"
        else:
            colour = "b"
        #create the new belief state
        new_belief_state = BeliefState(
            initial_beliefs=deepcopy(self.beliefs),
            agent_colour=self.agent_colour,
            max_depth=self.max_depth - 1
        )
        #play the move
        new_belief_state.update_information(action[0], action[1], colour)
        return new_belief_state


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
        #TODO use (virtual) connected components to improve utility function
