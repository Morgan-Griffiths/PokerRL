import numpy as np
from pokerrl_env.config import Config
from pokerrl_env.datatypes import BOARD_CARDS_GIVEN_STREET, SEAT_TO_POSITION, INT_TO_STREET
from pokerrl_env.utils import human_readable_cards


def return_board_cards(global_state, config):
    num_board_cards = BOARD_CARDS_GIVEN_STREET[ global_state[config.global_state_mapping["street"]] ]
    if num_board_cards == 0:
        padded_board_cards = np.zeros(10)
    else:
        board_slice = global_state[ config.global_state_mapping["board_range"][ 0 ] : config.global_state_mapping["board_range"][1] ][:num_board_cards]
        padded_array = np.zeros(10 - num_board_cards)
        padded_board_cards = np.concatenate((board_slice,padded_array))
    return padded_board_cards


def convert_global_state_to_player_view(global_state, player_index, config):
    pmap = config.player_state_mapping
    gmap = config.global_state_mapping
    player_state = np.zeros(config.player_state_shape)
    vil_index = lambda x: x
    for i,position in enumerate(config.player_positions,start=1):

        # match player position
        if player_index == position:
            player_state[pmap["hand_range"][0] : pmap["hand_range"][1]] = global_state[ gmap[f"player_{position}_hand_range"][0] : gmap[f"player_{position}_hand_range"][1] ]
            player_state[pmap["hero_active"]] = global_state[gmap[f"player_{position}_active"]]
            player_state[pmap["hero_stack"]] = global_state[gmap[f"player_{position}_stack"]]
            player_state[pmap["hero_position"]] = global_state[gmap[f"player_{position}_position"]]
            vil_index = lambda x: x - 1
        else:
            player_state[pmap[f"vil_{vil_index(i)}_active"]] = global_state[ gmap[f"player_{position}_active"] ]
            player_state[pmap[f"vil_{vil_index(i)}_stack"]] = global_state[ gmap[f"player_{position}_stack"] ]
            player_state[pmap[f"vil_{vil_index(i)}_position"]] = global_state[ gmap[f"player_{position}_position"] ]
    # Copy over the board cards
    player_state[pmap["board_range"][0] : pmap["board_range"][1]] = return_board_cards( global_state,config )
    global_state[gmap["board_range"][0]]: global_state[gmap["board_range"][1]]
    # copy general state
    for key in [
        "pot",
        "amount_to_call",
        "pot_odds",
        "street",
        "num_players",
        "current_player",
        "previous_amount",
        "previous_position",
        "previous_action",
        "previous_bet_is_blind",
        "last_agro_amount",
        "last_agro_position",
        "last_agro_action",
        "last_agro_bet_is_blind",
        "next_player",
    ]:
        player_state[pmap[key]] = global_state[gmap[key]]
    return player_state


def player_view(global_states, player_index, config):
    assert player_index in list(range(1,7)), "Player index must be between 1 and 6"
    # assert global_states[-1, config.global_state_mapping["current_player"]] == player_index, "Player index does not match current player"
    player_states = [
        convert_global_state_to_player_view(global_states[i,:], player_index, config)
        for i in range(global_states.shape[0])
    ]
    return np.stack(player_states)

def flatten(data:list):
    return [item for sublist in data for item in sublist]

def display_readable_states(readable_states):
    print("Hand : ","".join(flatten(readable_states[-1]['hand_range'])))
    for state in readable_states:
        print(state)
        print("Board: ","".join(flatten(state['board_range'])))
        print("Pot: ",state['pot'])
        print("hero_stack: ",state['hero_stack'])
        print("vil_1_stack: ",state['vil_1_stack'])
        print("vil_2_stack: ",state['vil_2_stack'])
        print("last_agro_position: ",state['last_agro_position'])
        print("last_agro_amount: ",state['last_agro_amount'])
        print("last_agro_action: ",state['last_agro_action'])
        print("previous_amount: ",state['previous_amount'])
        print("previous_position: ",state['previous_position'])
        print("previous_action: ",state['previous_action'])

def human_readable_view(global_states, player_index, config,display=False):
    player_states = player_view(global_states, player_index, config)
    readable_states = []
    for player_state in player_states:
        human_readable = {}
        for key, value in config.player_state_mapping.items():
            if "hand_range" in key:
                cards = human_readable_cards(player_state[value[0] : value[1]])
                human_readable[key] = cards
            elif "_position" in key:
                human_readable[key] = SEAT_TO_POSITION[int(player_state[value])]
            elif "_action" in key:
                human_readable[key] = config.action_to_str[int(player_state[value])]
            elif "street" in key:
                human_readable[key] = INT_TO_STREET[int(player_state[value])]
            elif (
                "_stack" in key
                or "_active" in key in key
                or "num_players" in key
                or "pot" in key
                or "amount_to_call" in key
                or "pot_odds" in key
                or "previous_amount" in key
                or "previous_bet_is_blind" in key
                or "previous_amount" in key
                or "previous_bet_is_blind" in key
                or "next_player" in key
                or "last_agro_amount" in key
                or "last_agro_position" in key
                or "last_agro_action" in key
                or "last_agro_bet_is_blind" in key
            ):
                human_readable[key] = player_state[value]
            elif "board_range" in key:
                cards = human_readable_cards(player_state[value[0] : value[1]])
                human_readable[key] = cards
        readable_states.append(human_readable)
    if display:
        display_readable_states(readable_states)
    return readable_states

def json_view(global_states, player_index, config:Config):
    player_states = player_view(global_states, player_index, config)
    json_states = []
    for state in player_states:
        state_object = {
            'hero_cards'                :state[config.player_state_mapping['hand_range'][0]:config.player_state_mapping['hand_range'][1]].tolist(),
            'board_cards'               :state[config.player_state_mapping['board_range'][0]:config.player_state_mapping['board_range'][1]].tolist(),
            'street'                    :state[config.player_state_mapping['street']],
            'num_players'               :state[config.player_state_mapping['num_players']],
            'hero_position'             :state[config.player_state_mapping['hero_position']],
            'hero_active'               :state[config.player_state_mapping['hero_active']],
            'vil1_active'               :state[config.player_state_mapping['vil_1_active']],
            'vil2_active'               :state[config.player_state_mapping['vil_2_active']],
            'vil3_active'               :state[config.player_state_mapping['vil_3_active']],
            'vil4_active'               :state[config.player_state_mapping['vil_4_active']],
            'vil5_active'               :state[config.player_state_mapping['vil_5_active']],
            'vil1_position'             :state[config.player_state_mapping['vil_1_position']],
            'vil2_position'             :state[config.player_state_mapping['vil_2_position']],
            'vil3_position'             :state[config.player_state_mapping['vil_3_position']],
            'vil4_position'             :state[config.player_state_mapping['vil_4_position']],
            'vil5_position'             :state[config.player_state_mapping['vil_5_position']],
            'last_agro_amount'          :state[config.player_state_mapping['last_agro_amount']],
            'last_agro_action'          :state[config.player_state_mapping['last_agro_action']],
            'last_agro_position'        :state[config.player_state_mapping['last_agro_position']],
            'last_agro_is_blind'        :state[config.player_state_mapping['last_agro_bet_is_blind']],
            'hero_stack'                :state[config.player_state_mapping['hero_stack']],
            'vil1_stack'                :state[config.player_state_mapping['vil_1_stack']],
            'vil2_stack'                :state[config.player_state_mapping['vil_2_stack']],
            'vil3_stack'                :state[config.player_state_mapping['vil_3_stack']],
            'vil4_stack'                :state[config.player_state_mapping['vil_4_stack']],
            'vil5_stack'                :state[config.player_state_mapping['vil_5_stack']],
            'pot'                       :state[config.player_state_mapping['pot']],
            'amount_to_call'            :state[config.player_state_mapping['amount_to_call']],
            'pot_odds'                  :state[config.player_state_mapping['pot_odds']],
            'previous_amount'           :state[config.player_state_mapping['previous_amount']],
            'previous_position'         :state[config.player_state_mapping['previous_position']],
            'previous_action'           :state[config.player_state_mapping['previous_action']],
            'previous_bet_is_blind'     :state[config.player_state_mapping['previous_bet_is_blind']],
            "current_player"            :state[config.player_state_mapping['current_player']],
            "next_player"               :state[config.player_state_mapping['next_player']],
        }
        json_states.append(state_object)
    return json_states