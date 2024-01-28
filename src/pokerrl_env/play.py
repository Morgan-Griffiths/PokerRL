from pokerrl_env.config import Config
from pokerrl_env.datatypes import StateActions
from pokerrl_env.utils import return_current_player
from pokerrl_env.view import player_view, human_readable_view
from pokerrl_env.transition import step_state,init_state

def get_action(action_mask,global_states,config):
    readable_actions = ['FOLD','CHECK','CALL']
    if global_states[-1,config.global_state_mapping['last_agro_action']] > StateActions.CALL:
        agro_action = 'RAISE'
    else:
        agro_action = 'BET'
    for betsize in config.betsizes:
        readable_actions.append(f'{agro_action} {betsize}')
    action_dict = {action:i for i,action in enumerate(readable_actions)}
    while True:
        possible_actions = [action for i,action in enumerate(readable_actions) if action_mask[i] > 0]
        try:
            print(list(range(len(possible_actions))))
            action_str = int(input(f'Enter the index of the desired action: {possible_actions}:'))
            return action_dict[possible_actions[action_str]]
        except ValueError:
            print('invalid action')

def play_game():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    while not done:
        player_idx = return_current_player(global_state)
        human_readable_view(global_state,player_idx, config,display=True)
        action = get_action(action_mask,global_state,config)
        global_state,done,winnings,action_mask = step_state(global_state, action, config)
    human_readable_view(global_state,global_state[-1,config.global_state_mapping['current_player']], config,display=True)
    print('winnings',winnings)

if __name__ == "__main__":
    play_game()