from pokerrl_env.transition import init_state, step_state
from pokerrl_env.config import Config
from pokerrl_env.utils import return_current_player
from pokerrl_env.view import player_view,json_view

class Game:
    def __init__(self,config:Config):
        self.config = config
        self.global_state = None
        self.done = None
        self.winnings = None
        self.action_mask = None

    def reset(self):
        """ Returns (state, reward, done, info) from the current player's perspective """
        self.global_state, self.done, self.winnings, self.action_mask = init_state(self.config)
        current_player = return_current_player(self.global_state,self.config)
        if self.config.is_server:
            return {"game_states":json_view(self.global_state,current_player,self.config), "done":self.done, "winnings":self.winnings, "action_mask":self.action_mask.tolist()}
        return self.global_state, self.done, self.winnings, self.action_mask

    def step(self,action):
        """ Returns (state, reward, done, info) from the current player's perspective """
        self.global_state, self.done, self.winnings, self.action_mask = step_state(self.global_state, action, self.config)
        current_player = return_current_player(self.global_state,self.config)
        if self.config.is_server:
            return {"game_states":json_view(self.global_state,current_player,self.config), "done":self.done, "winnings":self.winnings, "action_mask":self.action_mask.tolist()}
        return self.global_state, self.done, self.winnings, self.action_mask