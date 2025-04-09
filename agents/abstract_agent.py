"""
Module that includes an agent that has an abstracted perfect view of the board,
The imperfect aspect is included using a belief system that the agent never sees.
"""

from collections.abc import Callable
from pgmpy.models import FunctionalBayesianNetwork
from pgmpy.factors.hybrid import FunctionalCPD
import pyro.distributions as distribution
from agents.agent import Agent
import game.util as util

class AbstractAgent(Agent):
    """
    Agent that utilizes an advanced Hex AI and belief states
    To play Dark Hex
    """
    def __init__(self, num_cols: int, num_rows: int, settings: dict):
        """
        Create an abstract agent
        """
        super().__init__(num_cols, num_rows, settings)
        # assign all settings here
        self.cell_cpds : dict[tuple[int,int], FunctionalCPD] = None
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0)
        )
        self.cell_networks : dict[tuple[int,int], FunctionalBayesianNetwork] = None
        self.initialise_networks()
        self.weightings : dict[str, int] = settings["weightings"]
        self.opponents_unseen = 0
        self.empty_cells = num_cols * num_rows


    def move(self):
        """
        Make a move by the agent. 
        We search for the correct move in the most likely states
        and return the one with the highest expected gain
        """
        # first simulate the bayesian networks to find most likely places of their cells
        cell_count = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # simulate network for 100 samples
                sim_outputs = self.cell_networks[(col, row)].simulate(n_samples=100)
                cell_count.update({(col,row): (sim_outputs[(col, row)] == 1).sum()})
        # for now, we just place cells in the locations that were most common
        # TODO further rounds of simulation?
        guessed_cells = []
        for _ in range(self.opponents_unseen):
            max_cell = max(cell_count, key=cell_count.get)
            guessed_cells.append(max_cell)
            del cell_count[max_cell]
        fake_board = self.fake_board(guessed_cells)
        # run Hex AI on the fake board


    def update_information(self, col: int, row: int, colour: str):
        """
        Update our belief system with the new information
        """
        move_again = super().update_information(col, row, colour)
        if colour != self.colour:
            # we know (col, row) contains their colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(1))
            # we found one of their cells
            self.opponents_unseen -= 1
        else:
            # we know (col, row) contains our colour
            self.update_cell_cpd(col, row, distribution.Bernoulli(0))
            # the opponent will play somewhere that we haven't seen it
            self.opponents_unseen += 1
            self.empty_cells -= 2
        self.set_cell_cpds(
            cells=self._empty_cells(),
            dist=distribution.Bernoulli(self.opponents_unseen / self.empty_cells)
        )
        #TODO find a better distribution for the cell cpds
        return move_again


    def reset(self):
        self.set_cell_cpds(
            cells=[(x,y) for x in range(self.num_cols) for y in range(self.num_rows)],
            dist=distribution.Bernoulli(0)
        )
        self.initialise_networks()
        return super().reset()


    def set_cell_cpds(self, cells: list[tuple[int, int]], dist: distribution):
        """
        Set a list of cell cpds to be the same distribution
        """
        for cell in cells:
            self.update_cell_cpd(cell[0], cell[1], dist)


    def update_cell_cpd(self, col: int, row: int, dist: distribution):
        """
        Updates a single cell cpd to be the new distribution. 
        Resets the relevant cell cpd in all models it is present.
        """
        new_cpd = FunctionalCPD((col,row), dist)
        for cell in self._other_cells(col, row):
            self.cell_networks[cell].remove_cpds(self.cell_cpds[(col,row)])
            self.cell_networks[cell].add_cpds(new_cpd)
        self.cell_cpds[(col,row)] = new_cpd


    def initialise_networks(self):
        """
        Create the Bayesian Networks that contain our beliefs.
        Requires that the cell cpds have been set
        """
        def cell_exists(cell) -> bool:
            """
            Check whether a potential cell exists
            
            Parameters:
                cell: tuple[int, int] - The cell that we are checking the existence of.
            """
            return cell[0] >= 0 and cell[1] >= 0 and cell[0] < self.num_cols and cell[1] < self.num_rows

        def add_cond(cond: str, cells: list[tuple[int, int]]) -> tuple[(Callable), set[int], int]:
            """
            Adds a condition to the conditional probability distribution we are creating. 
            All elements returned should be combined with the currently maintained values.
            
            Parameters:
                cond: str - The condition we will be adding.
                cells: list[tuple[int, int]] - The cells that the condition includes
                
            Returns:
                fn: function[dict, int] - The function after adding the condition.
                parents: set[int] - The parents used in this condition.
                cond_weightings : int - The sum of weightings from this condition.
            """
            cond_weighting = 0
            parents = set()
            fn = lambda parent: 0
            # for each cell valid in this condition
            for cell in cells:
                if cell_exists(cell):
                    # add the cell to the function
                    cond_weighting += self.weightings[cond]
                    parents.add(cell)
                    fn = lambda parent: fn(parent) + self.weightings[cond] * parent[cell]
                    # note this may result in too much recursion?
            return (fn, parents, cond_weighting)

        # note here top left playable cell is (0,0), (col, row)

        conditions = [
            "width_1", "width_2_vc", "width_2_semi_vc"
        ]
        valid_cells = {
            "width_1": lambda col, row: [
                (col-1, row), (col, row-1), (col+1, row-1),
                (col+1, row), (col, row+1), (col-1,row+1)
            ],
            "width_2_vc": lambda col, row: [
                (col+1, row-2), (col+2, row-1), (col+1, row+1),
                (col-1, row+1), (col-2, row+1), (col-1, row-1)
            ],
            "width_2_semi_vc": lambda col, row: [
                (col+2, row-1), (col+2, row), (col, row+2),
                (col-2, row+2), (col-2, row), (col, row-2)
            ]
        }

        assert self.cell_cpds is not None

        def init_fn(parent) -> int:
            return 0

        # create a network centered around each node
        cell_networks = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # first create the cpd for this cell
                total_weighting = 0
                current_parents = set()
                current_fn = init_fn
                # determine distribution based on some conditions
                for cond in conditions:
                    if self.weightings[cond] > 0:
                        # add weightings of all cells within 1 distance
                        new_fn, new_parents, new_weighting = add_cond(
                            cond = cond,
                            cells = valid_cells[cond](col, row)
                        )
                        def updated_fn(parent):
                            current_fn(parent) + new_fn(parent)
                        current_fn = updated_fn
                        total_weighting += new_weighting
                        current_parents.union(new_parents)
                cpd = FunctionalCPD(
                    variable=(col, row),
                    #for now, we use a beta distribution with
                    # alpha = (weightings * present values)^2
                    # beta = total weighting
                    fn=lambda parent: distribution.Beta(current_fn(parent)**2, total_weighting),
                    parents=list(current_parents)
                )
                #now we create the network for this cell
                network_edges = self._other_cells(col, row)
                model = FunctionalBayesianNetwork(network_edges)
                model.add_cpds(cpd)
                cell_cpds = self._other_cells(col, row)
                model.add_cpds(cell_cpds)
                cell_networks.update({(col,row): model})


    def fake_board(self, cells: list[tuple[int, int]]) -> list[list[str]]:
        """
        Creates a fake board based on what we know and extra guessed cells
        """
        new_board = [row.copy() for row in self.board]
        for cell in cells:
            new_board[cell[1]][cell[0]] = util.swap_colour(self.colour)
        return new_board


    def _empty_cells(self) -> list[tuple[int, int]]:
        """
        Return a list of the empty cells on the board
        """
        return [
            (x,y) for x in range(self.num_cols) for y in range(self.num_rows) if self.board[y][x] == 'e'
        ]


    def _other_cells(self, col: int, row: int) -> list[tuple[int, int]]:
        """
        Return a list of all cells other than the input one
        """
        return [
            self.cell_cpds[y][x] for x in range(self.num_cols) for y in range(self.num_rows) if x!=col and y!=row
        ]


# For the hex AI, we require some way of returning value of the move,
# ideally the value of multiple moves to add to a large expected value
# accumulation, so we can calculate the true best expected value of a move

class HexAgent():
    """
    Class for the base Hex AI that the abstracted agent uses.
    """
    def __init__(self, agent_colour: str):
        pass


    def best_move(self, board: list[list[str]]) -> tuple[int, int]:
        """
        Search for the best move in a given board.
        Returns:
            move: tuple[int, int] - The chosen best move. The top left cell is (0,0), (col, row).
        """


    def board_scores(self, board: list[list[str]]) -> list[list[int]]:
        """
        Find the expected values of playing in each cell within the board.
        Returns:
            board: list[list[int]] - The scores of every cell on the board.
        """
