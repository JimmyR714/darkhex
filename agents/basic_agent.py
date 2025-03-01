"""
Module for a very basic agent that
works solely off of the rules of dark hex
"""
from typing import Self
from copy import deepcopy
import math
import logging
from scipy.cluster.hierarchy import DisjointSet
import agents.agent
import game.util as util

class BasicAgent(agents.agent.Agent):
    """
    Basic agent that uses the rules of dark hex, but no advanced techniques.
    """
    MAX_DEPTH = 3
    MAX_BELIEFS = 20
    def __init__(self, num_cols: int, num_rows: int, settings: dict):
        """Create a basic agent"""
        super().__init__(num_cols, num_rows, settings)
        self.belief_state = BeliefState.fresh(
            num_cols=num_cols,
            num_rows=num_rows,
            agent_colour=settings["colour"],
            max_depth=self.MAX_DEPTH,
            max_beliefs=self.MAX_BELIEFS
        )


    def move(self):
        """
        Make a move by the agent.
        Moves are determined by the one that results in the belief state with the 
        highest probability of our success.
        """
        return self.belief_state.optimal_move()


    def update_information(self, col: int, row: int, colour: str):
        another_move = self.belief_state.update_information(col, row, colour)
        if not another_move:
            #allow opponent's move update
            self.belief_state.opponent_move()
        return another_move


class Belief():
    """
    A single belief within the belief state.
    Initially the empty board
    """
    def __init__(self, board: list[list[str]], probability: float,
                 white_components: DisjointSet, black_components: DisjointSet) -> None:
        self.board = board
        self.probability = probability
        self.white_components = white_components
        self.black_components = black_components


    @classmethod
    def fresh(cls, num_cols: int, num_rows: int):
        """
        Create a fresh belief
        """
        (white_components, black_components) = util.create_default_components(
            num_cols=num_cols,
            num_rows=num_rows
        )
        return cls(
            board = [["e"] * num_cols] * num_rows,
            probability = 1.0,
            white_components = white_components,
            black_components = black_components
        )


    @classmethod
    def from_belief(cls, old_belief: Self, prob_ratio: float):
        """
        Create a belief based on an old one. 
        Probability is multiplied by prob_ratio
        """
        return cls(
            board = old_belief.board,
            probability = old_belief.probability * prob_ratio,
            white_components = old_belief.white_components,
            black_components = old_belief.black_components
        )

    def update_information(self, col: int, row: int, colour: str, agent_colour: str) -> None:
        """
        Updates the board and components of this belief
        """
        if colour == agent_colour:
            #we successfully placed our piece, and it is unseen
            self.board[row][col] = agent_colour + "u"
        elif colour != agent_colour and self.board[row][col] == colour + "u":
            #we discovered one of their pieces,
            # hence remove all beliefs where this wasn't unseen
            self.board[row][col] = colour + "s"
        # update components
        logging.debug("Updating board at %s, %s", col, row)
        util.update_components(
            cell_pos=(col+1, row+1),
            board=self.board,
            components=(self.white_components, self.black_components),
            colour=colour,
            borders=False
        )


    def utility(self) -> float:
        """
        Calculates the utility of this belief.
        White is a positive utility, black is negative.
        """
        def component_strength(components: DisjointSet) -> float:
            """
            Calculates strength of the components
            """
            #TODO use (virtual) connected components to improve utility function
            total = 1.0
            #for now, just based on size of components
            for component in components:
                total += len(component)
            return total


        def seen_strength(colour: str) -> float:
            """
            Calculates strength of the seen elements on the board for colour
            """
            total = 1.0
            for row in self.board:
                for cell in row:
                    if "u" in cell and colour in cell:
                        #if they haven't seen our cell
                        total += 3
                    elif "s" in cell and util.swap_colour(colour) in cell:
                        #if we have seen their cell
                        total += 5
                    elif "u" in cell and util.swap_colour(colour) in cell:
                        #we haven't seen their cell
                        total -= 3
                    elif "s" in cell and colour in cell:
                        #they have seen our cell
                        total -= 5
            return total

        #start with win checks
        win_check = util.win_check(
            num_rows=len(self.board),
            num_cols=len(self.board[0]),
            white_components=self.white_components,
            black_components=self.black_components
        )
        if win_check == "none":
            #perform an ordinary evaluation
            #TODO use a similar evaluation function to HEXY
            total_utility = 0.0
            #first include component strength
            white_component_strength = component_strength(self.white_components)
            black_component_strength = component_strength(self.black_components)
            total_utility += math.tanh(
                white_component_strength / black_component_strength
            )
            #then include seen strength
            total_utility += math.tanh(
                seen_strength("w") / seen_strength("b")
            )
        elif win_check == "white_win":
            total_utility = 10000.0
        else:
            total_utility = -10000.0

        #return expected utility
        return total_utility * self.probability


class BeliefState():
    """
    Class that represents a belief state for the agent.
    Maintains a list of beliefs about our current state.
    Each belief is a full amount of information about the board, hence applying any
    action to a belief is deterministic based on that belief being true.
    """
    def __init__(self, beliefs: list[Belief], agent_colour: str,
                 max_depth: int, max_beliefs: int, placeable_cells: list[tuple[int,int]]):
        self.agent_colour = agent_colour
        self.num_rows = len(beliefs[0].board)
        self.num_cols = len(beliefs[0].board[0])
        self.max_depth = max_depth
        self.max_beliefs = max_beliefs
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
        self.beliefs = beliefs
        #we always believe the same spaces may be empty (or containing an unseen opponent)
        #irrespective of the list of beliefs. Hence we maintain this:
        self.placeable_cells = placeable_cells


    @classmethod
    def fresh(cls, num_cols: int, num_rows: int, agent_colour: str,
              max_depth: int, max_beliefs: int):
        """
        Define a fresh belief state
        """
        logging.debug("Creating belief state from fresh")
        placeable_cells : list[tuple[int, int]] = []
        for col in range(num_cols):
            for row in range(num_rows):
                placeable_cells.append((col, row))
        return cls(
            beliefs=[Belief.fresh(num_cols, num_rows)],
            agent_colour=agent_colour,
            max_depth=max_depth,
            max_beliefs=max_beliefs,
            placeable_cells=placeable_cells
        )


    @classmethod
    def from_state(cls, beliefs: list[Belief], placeable_cells: list[tuple[int, int]],
                   agent_colour: str, max_depth: int, max_beliefs: int):
        """
        Define a belief state from a previous one
        """
        logging.debug("Creating belief state from current state")
        return cls(
            beliefs=beliefs,
            agent_colour=agent_colour,
            max_depth=max_depth,
            max_beliefs=max_beliefs,
            placeable_cells=deepcopy(placeable_cells)
        )


    def opponent_move(self) -> None:
        """
        Performs the required updates to the beliefs after an opponent has performed a move.
        We take all of our current beliefs and include the possibilities for the opponent moving.
        Must be called every time the opponent has moved to allow us to change our beliefs.
        """
        #TODO each probability of move is assumed equal in a given belief. we should take account of
        #     the goals of the opponent, i.e. they should think like us in theory
        logging.debug("Calculating opponent move")
        new_beliefs = []
        #first check which cells they've found
        for belief in self.beliefs:
            # if a cell of our colour is unseen, they might have found it
            for y, row in enumerate(belief.board):
                for x, cell in enumerate(row):
                    if cell == "wu" and self.agent_colour == "w":
                        #TODO opponent can only find one piece per turn
                        new_belief = Belief.from_belief(belief, 0.5)
                        new_belief.update_information(x, y, "w", "b")
                        new_beliefs.append(new_belief)
                        new_beliefs.append(Belief.from_belief(belief, 0.5)) # update probability
                    elif cell == "bu" and self.agent_colour == "b":
                        #they might have found our piece
                        new_belief = Belief.from_belief(belief, 0.5)
                        new_belief.update_information(x, y, "b", "w")
                        new_beliefs.append(new_belief)
                        new_beliefs.append(Belief.from_belief(belief, 0.5)) # update probability
                    else:
                        new_beliefs.append(belief)
        #TODO bug when there are no placable cells, maybe fix by adding win condition?
        final_beliefs = []
        num_cells = len(self.placeable_cells)
        # if a cell is empty, they might have placed their piece there
        for belief in new_beliefs:
            final_beliefs.append(Belief.from_belief(belief, 1.0/num_cells))
            for cell in self.placeable_cells:
                new_belief = Belief.from_belief(belief, 1.0/num_cells)
                opp_colour = util.swap_colour(self.agent_colour)
                new_belief.update_information(cell[0], cell[1], opp_colour, opp_colour)
                final_beliefs.append(new_belief)
        self.beliefs = final_beliefs

        #reduce the number of beliefs
        self.reduce_beliefs()


    def ordered_move_list(self) -> list[tuple[int, int, float]]:
        """
        Returns the list of expected opponent moves from most probable to least.
        Uses the min/max search to find the value of each action and then assign
        probabilities based on action value.
        """
        #TODO, implement this to allow opponent probabilities. may need to store opponent beliefs?


    def optimal_move(self) -> tuple[int, int]:
        """
        Returns the optimal move in the current belief state
        Uses alpha-beta pruned minimax search
        """
        # start the minimax search
        white_turn = self.agent_colour=="w"
        best_action = None
        best_utility = None
        for action in self.placeable_cells:
            result = self.result(white_turn, action)
            utility_value = result.min_value(
                -1000000000.0, 1000000000.0, white_turn
            )
            if best_utility is None or utility_value > best_utility:
                best_action = action
                best_utility = utility_value
        #TODO bug where this returns None on a 2x1 board
        return best_action


    def min_value(self, alpha: float, beta: float, white_turn: bool) -> float:
        """
        Returns the minimum expected utility value in state
        """
        logging.debug(
            "Finding minimum value for %s beliefs, alpha=%s, beta=%s, white_turn=%s",
            len(self.beliefs),
            alpha,
            beta,
            white_turn
        )
        # check for termination
        if self.terminal_test():
            logging.debug("Terminal test was successful")
            return self.utility()
        # search for min value
        current_min_value = 1000000000.0
        # check each possible action
        logging.debug("Checking each action for min value")
        for action in self.placeable_cells:
            # try this action, update min value if necessary
            result = self.result(white_turn, action)
            result_value = result.max_value(
                alpha,
                beta,
                white_turn=not white_turn
            )
            current_min_value = min(
                current_min_value,
                result_value
            )
            # attempt pruning
            if current_min_value <= alpha:
                logging.debug("Pruned by min value vs alpha")
                return current_min_value
            #update pruning value
            beta = min(beta, current_min_value)

        return current_min_value


    def max_value(self, alpha: float, beta: float, white_turn: bool) -> float:
        """
        Returns the maximum expected utility value in state
        """
        logging.debug(
            "Finding maximum value for %s beliefs, alpha=%s, beta=%s, white_turn=%s",
            len(self.beliefs),
            alpha,
            beta,
            white_turn
        )
        # check for termination
        if self.terminal_test():
            logging.debug("Terminal test was successful")
            return self.utility()
        # search for max value
        current_max_value = -1000000000.0
        # check each possible action
        logging.debug("Checking each action for max value")
        for action in self.placeable_cells:
            # try this action, update max value if necessary
            result = self.result(white_turn, action)
            result_value = result.min_value(
                alpha,
                beta,
                white_turn=not white_turn
            )
            current_max_value = max(
                current_max_value,
                result_value
            )
            # attempt pruning
            if current_max_value >= beta:
                logging.debug("Pruned by max value vs beta")
                return current_max_value
            # update pruning value
            alpha = max(alpha, current_max_value)

        return current_max_value


    def update_information(self, col: int, row: int, colour: str) -> bool:
        """
        Receive some updated information about the board.
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
        logging.debug("Updating belief state with move (%s, %s), %s", col, row, colour)
        for belief in self.beliefs:
            belief.update_information(col, row, colour, self.agent_colour)

        #we can no longer place a piece in this cell
        self.placeable_cells.remove((col, row))

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
        new_belief_state = BeliefState.from_state(
            beliefs=deepcopy(self.beliefs),
            agent_colour=self.agent_colour,
            max_depth=self.max_depth-1,
            max_beliefs=self.max_beliefs,
            placeable_cells=self.placeable_cells
        )
        #play the move
        new_belief_state.update_information(col=action[0], row=action[1], colour=colour)
        logging.debug("Belief state created")
        return new_belief_state


    def terminal_test(self) -> bool:
        """
        Determine whether we should terminate the search
        """
        #TODO add win check
        logging.debug(("Terminal test with max depth", self.max_depth))
        #check whether we have reached the max depth
        if self.max_depth == 0:
            return True
        #check whether a player has won in any belief
        winner = False
        for belief in self.beliefs:
            winner = winner or "none" != util.win_check(
                num_rows=self.num_rows,
                num_cols=self.num_cols,
                white_components=belief.white_components,
                black_components=belief.black_components
            )
        return winner


    def reduce_beliefs(self) -> None:
        """
        Reduces beliefs so that we only maintain up to max_beliefs.
        Keep only the beliefs with the highest probability
        """
        #TODO maybe add probability threshold as alternative option
        self.beliefs = sorted(
            self.beliefs, key=lambda b: b.probability, reverse=True
        )[:self.max_beliefs]


    def utility(self) -> float:
        """
        The utility of the belief state
        """
        return sum([b.utility() for b in self.beliefs])
