from typing import Dict, List, Tuple
import numpy as np
from pokerrl.config import Config
from pokerrl.datatypes import PLAYER_ORDER_BY_STREET, POSITION_TO_SEAT,RAISE,CALL,FOLD,BET,CHECK, ModelActions, StateActions,Street,Player,Positions
from pokerrl.cardlib import encode, hand_rank
from pokerrl.utils import is_next_player_the_aggressor, return_deck
import copy

def init_state(config: Config):
    deck = return_deck()
    
    state_SB = np.zeros(config.global_state_shape) 
    state_BB = np.zeros(config.global_state_shape)
    first_player = 'Dealer' if config.num_players == 2 else 'Small Blind'

    for state in (state_SB, state_BB):
        state[config.global_state_mapping["num_players"]] = config.num_players
        state[config.global_state_mapping["pot"]] = sum(config.blinds)

    if isinstance(config.stack_sizes, int):
        stack_sizes = [config.stack_sizes] * config.num_players
    else:
        stack_sizes = config.stack_sizes

    for i,position in enumerate(config.player_positions):
        hand = np.array([*deck.pop(), *deck.pop(),*deck.pop(), *deck.pop()],dtype=np.uint8)
        for state in (state_SB, state_BB):
            # Set player positions
            state[config.global_state_mapping[f"player_{position}_position"]] = position
            # Set player stack sizes
            state[config.global_state_mapping[f"player_{position}_stack"]] = stack_sizes[i]
            # Set player active status
            state[config.global_state_mapping[f"player_{position}_active"]] = 1
            # Set player hole cards
            state[config.global_state_mapping[f"player_{position}_hand_range"][0]:config.global_state_mapping[f"player_{position}_hand_range"][1]] = hand
    
    # board cards
    board_cards = np.array([*deck.pop(), *deck.pop(),*deck.pop(), *deck.pop(), *deck.pop()])

    # Small Blind action
    state_SB[config.global_state_mapping["current_player"]] = POSITION_TO_SEAT["Big Blind"]
    state_SB[config.global_state_mapping["previous_amount"]] = config.blinds[0]
    state_SB[config.global_state_mapping["previous_position"]] = POSITION_TO_SEAT[first_player]
    state_SB[config.global_state_mapping["previous_action"]] = 4  # Posting Small Blind is considered a raise
    state_SB[config.global_state_mapping["previous_bet_is_blind"]] = 1
    state_SB[config.global_state_mapping["last_agro_amount"]] = config.blinds[0]
    state_SB[config.global_state_mapping["last_agro_position"]] = POSITION_TO_SEAT[first_player]
    state_SB[config.global_state_mapping["last_agro_action"]] = StateActions.CALL + 1  # Posting Small Blind is considered a raise
    state_SB[config.global_state_mapping["last_agro_bet_is_blind"]] = 1
    state_SB[config.global_state_mapping["pot"]] = 0.5
    state_SB[config.global_state_mapping["next_player"]] = config.player_positions[1]
    state_SB[config.global_state_mapping["board_range"][0]:config.global_state_mapping["board_range"][1]] = board_cards
    state_SB[config.global_state_mapping["street"]] = Street.PREFLOP
    state_SB[config.global_state_mapping[f"player_{POSITION_TO_SEAT[first_player]}_stack"]] -= config.blinds[0]

    # Big Blind action
    state_BB[config.global_state_mapping["current_player"]] = config.player_positions[2 % config.num_players]
    state_BB[config.global_state_mapping["previous_amount"]] = config.blinds[1]
    state_BB[config.global_state_mapping["previous_position"]] = POSITION_TO_SEAT["Big Blind"]
    state_BB[config.global_state_mapping["previous_action"]] = StateActions.CALL + 1  # Posting Big Blind is considered a raise
    state_BB[config.global_state_mapping["previous_bet_is_blind"]] = 1
    state_BB[config.global_state_mapping["last_agro_amount"]] = config.blinds[1]
    state_BB[config.global_state_mapping["last_agro_position"]] = POSITION_TO_SEAT["Big Blind"]
    state_BB[config.global_state_mapping["last_agro_action"]] = 4  # Posting Small Blind is considered a raise
    state_BB[config.global_state_mapping["last_agro_bet_is_blind"]] = 1
    state_BB[config.global_state_mapping["pot"]] = 1.5
    state_BB[config.global_state_mapping["next_player"]] = config.player_positions[3 % config.num_players]
    state_BB[config.global_state_mapping["board_range"][0]:config.global_state_mapping["board_range"][1]] = board_cards
    state_BB[config.global_state_mapping["street"]] = Street.PREFLOP
    state_BB[config.global_state_mapping["player_2_stack"]] -= config.blinds[1]
    state_BB[config.global_state_mapping[f"player_{POSITION_TO_SEAT[first_player]}_stack"]] -= config.blinds[0]

    winnings = {position: 0 for position in config.player_positions}
    done = False
    player_totals = {position: 0 for position in config.player_positions}
    player_totals[POSITION_TO_SEAT[first_player]] = config.blinds[0]
    player_totals[POSITION_TO_SEAT["Big Blind"]] = config.blinds[1]
    return np.stack((state_SB, state_BB), axis=0),done,winnings,get_action_mask(state_BB,player_totals,config)

def clear_last_agro_action(global_state,config:Config):
    global_state[config.global_state_mapping[f'last_agro_action']] = 0
    global_state[config.global_state_mapping[f'last_agro_position']] = 0
    global_state[config.global_state_mapping[f'last_agro_amount']] = 0
    global_state[config.global_state_mapping[f'last_agro_bet_is_blind']] = 0

def players_finished(global_state,config:Config):
    num_active_players = 0
    for position in config.player_positions:
        if global_state[config.global_state_mapping[f'player_{position}_active']] == 1 and global_state[config.global_state_mapping[f'player_{position}_stack']] > 0:
            num_active_players += 1
            if num_active_players > 1:
                return False
    return True

def classify_action(action,player_street_total,player_stack,last_agro_amount,last_agro_action,pot,config:Config) -> Tuple[ModelActions, int]: 
    """ Returns action string and associated betsize """
    if action == ModelActions.FOLD:
        return FOLD, 0
    elif action == ModelActions.CHECK:
        return CHECK, 0
    elif action == ModelActions.CALL:
        return CALL, last_agro_amount - player_street_total
    elif action > ModelActions.CALL:
        # either bet or raise.
        return config.return_betsize(last_agro_action,last_agro_amount,config,action,pot,player_street_total,player_stack)
    else:
        raise ValueError(f"Invalid action: {action}")

def order_players_by_street(global_state:np.ndarray,config:Config):
    active_players = []
    for i in range(1,7):
        if global_state[config.global_state_mapping[f'player_{i}_stack']] > 0 and global_state[config.global_state_mapping[f'player_{i}_active']] == 1:
            active_players.append(Player(global_state[config.global_state_mapping[f'player_{i}_position']],global_state[config.global_state_mapping[f'player_{i}_stack']],global_state[config.global_state_mapping[f'player_{i}_active']]))
    player_ordering = PLAYER_ORDER_BY_STREET[int(global_state[config.global_state_mapping['street']])]
    active_players.sort(key=lambda x: player_ordering[x.position])
    return active_players

def get_pots(pot_players: List[Player], total_amount_invested: Dict[int, float]) -> List[Tuple[float, List[Player]]]:
    pots = []
    while pot_players:
        num_players = len(pot_players)
        min_invested = total_amount_invested[min(pot_players, key=lambda x: total_amount_invested[x.position]).position]
        # print('min_invested',min_invested)
        pot = 0
        involved_players = []

        # print('player',pot_players)
        players_to_remove = []
        for player in pot_players:
            involved_players.append(player)
            pot += min_invested
            total_amount_invested[player.position] -= min_invested
            if total_amount_invested[player.position] == 0:
                players_to_remove.append(player)
        for player in players_to_remove:
            pot_players.remove(player)
        pots.append([pot, involved_players])
        # print('pots,pot_players',pots,len(pot_players))
        if num_players == len(pot_players):
            break
    # print('total_amount_invested',total_amount_invested)
    if sum(total_amount_invested.values()) > 0:
        pots[-1][0] += sum(total_amount_invested.values())
    return pots

def calculate_winnings(pots: List[Tuple[float, List[Player]]], players: List[Player]) -> Dict[int, float]:
    winnings = {}
    for p in players:
        winnings[p.position] = {
            'hand': p.hand if p.active else [],
            'hand_value': p.hand_value if p.active else 0,
            'result': -p.total_invested
        }
    for pot, involved_players in pots:
        min_hand_rank = min(involved_players, key=lambda x: x.hand_value).hand_value
        # print('min_hand_rank',min_hand_rank)
        winners = [player for player in involved_players if player.hand_value == min_hand_rank]
        # print('winners',winners)
        player_winnings = pot / len(winners)
        # print('player_winnings',player_winnings)
        for winner in winners:
            winner.stack += player_winnings
            winnings[winner.position]['result'] += player_winnings
    # print('winnings',winnings)
    return winnings

def game_over(global_state:np.ndarray,config:Config,total_amount_invested:float):
    # get all hand values
    # print('game_over')
    # print('pot',global_state[config.global_state_mapping['pot']])
    # print('total_amount_invested',total_amount_invested)
    board = global_state[config.global_state_mapping[f'board_range'][0]:config.global_state_mapping[f'board_range'][1]]
    board = [int(h) for h in board]
    en_board = [encode(board[i*2:(i*2)+2]) for i in range(0,len(board)//2)]
    players = []
    for position in config.player_positions:
        # print('position',position)
        hand_start = config.global_state_mapping[f'player_{position}_hand_range'][0]
        hand_end = config.global_state_mapping[f'player_{position}_hand_range'][1]
        player_hand = [int(h) for h in global_state[hand_start:hand_end]]
        encoded_hand = [encode(player_hand[i*2:(i*2)+2]) for i in range(0,len(player_hand)//2)]
        hand_value = hand_rank(encoded_hand, en_board)
        players.append(Player(position=global_state[config.global_state_mapping[f'player_{position}_position']],
                                    stack=global_state[config.global_state_mapping[f'player_{position}_stack']],
                                    active=global_state[config.global_state_mapping[f'player_{position}_active']],
                                    hand=player_hand,
                                    hand_value=hand_value,
                                    total_invested=total_amount_invested[position]))
    
    print('players',players)
    pot_players = [p for p in players if p.active > 0]
    print('pot_players',pot_players)
    if len(pot_players) > 1:
        # accelerate street to river
        global_state[config.global_state_mapping['street']] = 4
    # Identify side pots and main pot
    pots = get_pots(pot_players, total_amount_invested)
    print('pots',pots)
    # Find winners and distribute the pots
    winnings = calculate_winnings(pots, players)
    # print('total winnings',winnings)
    return winnings

def increment_players(global_state:np.ndarray,active_players:list,current_player:int,config:Config):
    """ Skip players with stack 0. Which can happen if a player when allin but there are 2+ active players remaining """
    global_state[config.global_state_mapping['current_player']] = global_state[config.global_state_mapping['next_player']]
    non_zero_players = [p for p in active_players if p.stack > 0]
    player_idx = [p.position for p in non_zero_players].index(current_player)
    # print('player_idx',player_idx)
    for player in non_zero_players:
        if player.position == current_player:
            next_player = non_zero_players[(player_idx + 2) % len(non_zero_players)]
    global_state[config.global_state_mapping[f'next_player']] = next_player.position

def new_street_player_order(global_state:np.ndarray,config:Config):
    """ Skip players with stack 0. Which can happen if a player when allin but there are 2+ active players remaining """
    active_players = order_players_by_street(global_state,config)
    non_zero_players = [p for p in active_players if p.stack > 0]
    global_state[config.global_state_mapping[f'current_player']] = non_zero_players[0].position if len(non_zero_players) > 0 else 0
    global_state[config.global_state_mapping[f'next_player']] = non_zero_players[1].position if len(non_zero_players) > 1 else 0

def get_action_mask(global_state, player_amount_invested_per_street, config:Config):
    current_player_position = int(global_state[config.global_state_mapping["current_player"]])
    if current_player_position > 0:
        current_player_stack = global_state[config.global_state_mapping[f"player_{current_player_position}_stack"]]
        if current_player_stack > 0:
            pot = global_state[config.global_state_mapping[f"pot"]]
            current_player_investment = player_amount_invested_per_street[current_player_position]
            return config.return_action_mask(global_state,config,pot,current_player_investment,current_player_stack)
    return np.zeros(config.num_actions)

def return_investments(global_states,config:Config):
    player_amount_invested_per_street = {position:0 for position in config.player_positions}
    player_total_amount_invested = {position:0 for position in config.player_positions}
    current_street = 1
    for global_state in global_states:
        previous_player = global_state[config.global_state_mapping['previous_position']]

        if global_state[config.global_state_mapping['street']] > current_street:
            # new street
            player_amount_invested_per_street = {position:0 for position in config.player_positions}
            current_street = global_state[config.global_state_mapping['street']]
            player_total_amount_invested[previous_player] += global_state[config.global_state_mapping[f'previous_amount']]
        else:
            if global_state[config.global_state_mapping[f'previous_action']] > StateActions.CALL and not global_state[config.global_state_mapping[f'previous_bet_is_blind']]:
                # Special case for raise
                player_total_amount_invested[previous_player] += global_state[config.global_state_mapping[f'previous_amount']] - player_amount_invested_per_street[previous_player]
                player_amount_invested_per_street[previous_player] += global_state[config.global_state_mapping[f'previous_amount']] - player_amount_invested_per_street[previous_player]
            else:
                player_total_amount_invested[previous_player] += global_state[config.global_state_mapping[f'previous_amount']]
                player_amount_invested_per_street[previous_player] += global_state[config.global_state_mapping[f'previous_amount']]
    return player_amount_invested_per_street, player_total_amount_invested

def step_state(global_states:np.ndarray, action:int, config:Config):
    """ Step the state forward by one action. Record the total amount invested by each player per street. """
    player_amount_invested_per_street, player_total_amount_invested = return_investments(global_states,config)
    # print street
    print('street',global_states[-1][config.global_state_mapping['street']])
    print('player_total_amount_invested',player_total_amount_invested)
    print('player_amount_invested_per_street',player_amount_invested_per_street)
    
    # calculate next state
    global_state = np.copy(global_states[-1])
    active_players = order_players_by_street(global_state,config)
    current_player = int(global_state[config.global_state_mapping['current_player']])
    print('current_player',current_player)
    print('active_players',active_players)
    # Get action details
    try:
        print('inputs','action',action,'player_amount_invested_per_street',player_amount_invested_per_street[current_player],'player_stack',global_state[config.global_state_mapping[f'player_{current_player}_stack']],'last_agro_amount',global_state[config.global_state_mapping['last_agro_amount']],'last_agro_action',global_state[config.global_state_mapping['last_agro_action']],'pot',global_state[config.global_state_mapping['pot']])
        action_category,betsize = classify_action(action, player_amount_invested_per_street[current_player], global_state[config.global_state_mapping[f'player_{current_player}_stack']], global_state[config.global_state_mapping['last_agro_amount']],global_state[config.global_state_mapping['last_agro_action']],global_state[config.global_state_mapping['pot']],config)
    except Exception as e:
        print('action',action)
        print('player_amount_invested_per_street',player_amount_invested_per_street[current_player])
        print('player_stack',global_state[config.global_state_mapping[f'player_{current_player}_stack']])
        print('last_agro_amount',global_state[config.global_state_mapping['last_agro_amount']])
        print('last_agro_action',global_state[config.global_state_mapping['last_agro_action']])
        print('pot',global_state[config.global_state_mapping['pot']])
        raise e
    print('action',action_category,betsize)
    print('player_amount_invested_per_street',player_amount_invested_per_street)
    player_total_amount_invested[current_player] += betsize
    global_state[config.global_state_mapping[f'pot']] += betsize
    print('prior player stack',global_state[config.global_state_mapping[f'player_{current_player}_stack']])
    if action_category == RAISE:
        global_state[config.global_state_mapping[f'player_{current_player}_stack']] -= betsize - player_amount_invested_per_street[current_player]
    else:
        global_state[config.global_state_mapping[f'player_{current_player}_stack']] -= betsize

    print('post player stack',global_state[config.global_state_mapping[f'player_{current_player}_stack']])
    global_state[config.global_state_mapping[f'previous_action']] = config.convert_model_action_to_state(action)
    global_state[config.global_state_mapping[f'previous_position']] = current_player
    global_state[config.global_state_mapping[f'previous_amount']] = betsize
    global_state[config.global_state_mapping[f'previous_bet_is_blind']] = 0
    done = False
    winnings = {position:{'hand_value':0,'hand':[],'result':0} for position in config.player_positions}
    # print('street',global_state[config.global_state_mapping[f'street']])
    if action_category in [BET, RAISE]:
        # Subtract the amount already invested.
        global_state[config.global_state_mapping[f'pot']] -= player_amount_invested_per_street[current_player]
        # update last agro action
        global_state[config.global_state_mapping[f'last_agro_action']] = config.convert_model_action_to_state(action)
        global_state[config.global_state_mapping[f'last_agro_position']] = current_player
        global_state[config.global_state_mapping[f'last_agro_amount']] = betsize
        global_state[config.global_state_mapping[f'last_agro_bet_is_blind']] = 0
        # update next player
        increment_players(global_state,active_players,current_player,config)

    elif action_category == CHECK:
        # print('check')
        if current_player == active_players[-1].position:
            # end of street
            if global_state[config.global_state_mapping[f'street']] == Street.RIVER:
                # end of game
                done = True
                winnings = game_over(global_state,config,player_total_amount_invested)
            else:
                global_state[config.global_state_mapping[f'street']] += 1
                clear_last_agro_action(global_state,config)
                new_street_player_order(global_state,config)
        else:
            increment_players(global_state,active_players,current_player,config)
    elif action_category == CALL:
        print('call')
        # print('call',current_player,global_state[config.global_state_mapping['next_player']],global_state[config.global_state_mapping['last_agro_position']])
        # Special case preflop blind situation.
        if global_state[config.global_state_mapping['street']] == Street.PREFLOP and \
            global_state[config.global_state_mapping['last_agro_amount']] == config.blinds[1] and \
            global_state[config.global_state_mapping['next_player']] == Positions.BIG_BLIND and \
            global_state[config.global_state_mapping[f'last_agro_position']] == Positions.BIG_BLIND:
            print('special case preflop blind situation')
            # bb can raise, or check
            increment_players(global_state,active_players,current_player,config)
        # or last aggressor is allin and would be the next player.
        # 4 allin 2,3 -> 4
        # 5 2 -> 4
        # 5 6 -> 4
        # (allin - current) % 6
        # current - vil = 1,-3,1
        elif global_state[config.global_state_mapping['next_player']] == global_state[config.global_state_mapping['last_agro_position']] or len(active_players) == 1 or global_state[config.global_state_mapping['last_agro_position']] not in active_players and is_next_player_the_aggressor(active_players,current_player,global_state[config.global_state_mapping['last_agro_position']]):
            print('next_player == last_agro_position',global_state[config.global_state_mapping['next_player']],global_state[config.global_state_mapping['last_agro_position']])
            # end of round. last street or all players allin
            if global_state[config.global_state_mapping['street']] == Street.RIVER or players_finished(global_state, config):
                # end of game
                done = True
                winnings = game_over(global_state,config,player_total_amount_invested)
            else:
                print('updating street')
                # update street
                global_state[config.global_state_mapping['street']] += 1
                clear_last_agro_action(global_state,config)
                new_street_player_order(global_state,config)
        else:
            print('incrementing players')
            print('next_player',global_state[config.global_state_mapping['next_player']])
            print('last_agro_position',global_state[config.global_state_mapping['last_agro_position']])
            increment_players(global_state,active_players,current_player,config)
    elif action_category == FOLD:
        print('current_player fold',current_player)
        global_state[config.global_state_mapping[f'player_{current_player}_active']] = 0
        # check for end of game
        if players_finished(global_state, config):
            # end game
            done = True
            winnings = game_over(global_state,config,player_total_amount_invested)
        elif global_state[config.global_state_mapping['next_player']] == global_state[config.global_state_mapping['last_agro_position']]:
            # end of round
            if global_state[config.global_state_mapping['street']] == Street.RIVER:
                # end of game
                winnings = game_over(global_state,config,player_total_amount_invested)
                done = True
            else:
                # update street
                global_state[config.global_state_mapping['street']] += 1
                clear_last_agro_action(global_state,config)
                new_street_player_order(global_state,config)
        else:
            increment_players(global_state,active_players,current_player,config)
    
    return np.concatenate([global_states,global_state[None,:]],axis=0),done,winnings,get_action_mask(global_state,player_amount_invested_per_street,config)