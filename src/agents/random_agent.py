import random
import numpy as np


class RandomPlayer:
    def __init__(self, args):
        self.args = args
        self.nb_action = len(args.action_names)

    def act(self, state):
        # TODO how to do if-else only once?

        if 'getout' == self.args.m:
            action = self.getout_actor()
        elif self.args.m == 'threefish':
            action = self.threefish_actor()
        elif self.args.m == 'loot':
            action = self.loot_actor()
        elif self.args.m == "Asterix":
            action = random.randint(0, self.nb_action - 1)
        else:
            raise ValueError
        return action

    def getout_actor(self):
        # action = coin_jump_actions_from_unified(random.randint(0, 10))
        return random.randint(0, 10)

    def threefish_actor(self):
        return np.random.randint([9])

    def loot_actor(self):
        return np.random.randint([9])
