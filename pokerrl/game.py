from pokerrl.transition import init_state, step_state
from pokerrl.config import Config
from pokerrl.utils import return_current_player
from pokerrl.view import player_view

class Game:
    def __init__(self,config):
        self.config = config
        self.global_state = None
        self.done = None
        self.winnings = None
        self.action_mask = None

    def reset(self):
        """ Returns (state, reward, done, info) from the current player's perspective """
        self.global_state, self.done, self.winnings, self.action_mask = init_state(self.config)
        return self.global_state, self.done, self.winnings, self.action_mask

    def step(self,action):
        """ Returns (state, reward, done, info) from the current player's perspective """
        self.global_state, self.done, self.winnings, self.action_mask = step_state(self.global_state, action, self.config)
        current_player = return_current_player(self.global_state)
        return player_view(self.global_state,current_player,self.config), self.done, self.winnings, self.action_mask