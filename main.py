from config import Config
from view import player_view, human_readable_view
from transition import step_state,init_state
import numpy as np

def main():
    CALL = 2
    CHECK = 1
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    readable = human_readable_view(global_state,1, config)
    global_state,done,winnings,action_mask = step_state(global_state, CALL, config)
    # print(readable)
    while not done:
        print('action_mask',action_mask)
        print('street',global_state[-1,config.global_state_mapping['street']])
        # action = np.random.choice(np.arange(config.num_actions), p=action_mask / np.sum(action_mask))
        # print('action',action)
        global_state,done,winnings,action_mask = step_state(global_state, CHECK, config)
        # readable = human_readable_view(global_state,1, config)
        # print(readable)
    print('street',global_state[:,config.global_state_mapping['street']])
    print('winnings',winnings)

if __name__ == "__main__":
    main()