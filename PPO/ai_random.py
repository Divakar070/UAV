'''
This module contains a random move.
'''

import random
from ai_base import RL

class RandomMove(RL):

    def __init__(self):
        super().__init__("Random-move")

    def execute(self, state, reward) -> int:
        '''It returns a random valid action.'''
        return random.choice(state.valid_actions())

    def load_data(self) -> int:
        print("- not applicable for this algorithm")
        return -1 # failed or NA

    def save_data(self, round) -> bool:
        print("- not applicable for this algorithm")
        return False # failed or NA
