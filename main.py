"""
Module for the control of the program flow
"""

import logging
import game.display as display
import game.darkhex as darkhex

class Controller:
    """
    Handles the high-level control over the abstract dark hex
    and the display window, as well as (in future), AI
    """
    def __init__(self):
        """
        Set constants necessary (maybe from a config file)
        """
        self.window = None
        self.game = None


    def make_window(self):
        """
        Create the display window for the program
        """
        self.window = display.DisplayWindow(self)
        self.window.mainloop()


    def new_game(self, num_cols, num_rows):
        """
        Create a new game of dark hex
        """
        self.game = darkhex.AbstractDarkHex(num_cols, num_rows)


def main():
    """
    Create the display for the game and run the mainloop
    """
    logging.basicConfig(level=logging.DEBUG)
    controller = Controller()
    controller.make_window()

if __name__ == "__main__":
    main()
