from typing import List
from pokerrl_env.datatypes import BET, RAISE, ModelActions, Player, StateActions, Street,Positions, int_to_rank, int_to_suit,rank_to_int,suit_to_int
from random import shuffle
import numpy as np

def return_deck():
    deck = []
    for suit in range(1,5):
        for rank in range(1,14):
            deck.append((rank,suit))
    shuffle(deck)
    return deck

def human_readable_cards(cards):
    readable_cards = []
    for i in range(0,len(cards),2):
        readable_cards.append([int_to_rank[cards[i]],int_to_suit[cards[i+1]]])
    return readable_cards

def readable_card_to_int(card):
    return [rank_to_int[card[0]],suit_to_int[card[1]]]

def return_current_player(global_states,config):
    return int(global_states[-1,config.global_state_mapping['current_player']])

def is_next_player_the_aggressor(active_players:List[Player], current_player:int, last_agro_position:int):
    """ for when the aggressor is allin and is not in active players. """
    current_player_index = [player.position for player in active_players].index(current_player)
    next_player_position = active_players[(current_player_index + 1) % len(active_players)].position
    if next_player_position < current_player:
        next_player_position + 6
    distance_to_agro = (last_agro_position - current_player) % 6
    distance_to_next = (next_player_position - current_player) % 6
    return distance_to_next > distance_to_agro
    

#### Action Mask Functions ####

def calculate_pot_limit_mask(global_state,config,pot,current_player_investment,current_player_stack):
    action_mask = np.zeros(config.num_actions, dtype=int)  # +2 for check and fold
    if global_state[config.global_state_mapping["last_agro_action"]] > ModelActions.CALL:
        max_raise = (pot - current_player_investment) + (2 * global_state[config.global_state_mapping["last_agro_amount"]])
        max_bet = min(current_player_stack+current_player_investment,max_raise)
        if max_bet <= global_state[config.global_state_mapping["last_agro_amount"]]:
            max_bet = 0

        # check for special case preflop blind situation.
        if global_state[config.global_state_mapping["street"]] == Street.PREFLOP and \
            global_state[config.global_state_mapping["current_player"]] == Positions.BIG_BLIND and \
            global_state[config.global_state_mapping["last_agro_position"]] == Positions.BIG_BLIND:
            action_mask[1] = 1
        else:
            action_mask[0] = 1 # fold is possible
            action_mask[2] = 1 # call is possible
    else:
        # bet
        max_bet = min(current_player_stack,pot)
        action_mask[1] = 1 # check is possible
    if max_bet > 0:
        highest_bet_ratio = current_player_stack / max_bet
        for i,bet_ratio in enumerate(config.betsizes):
            if bet_ratio <= highest_bet_ratio:
                action_mask[i + 3:] = 1
                break
    return action_mask


def calculate_no_limit_mask(global_state,config,pot,current_player_investment,current_player_stack):
    action_mask = np.zeros(config.num_actions, dtype=int)  # +2 for check and fold
    max_bet = current_player_stack
    if global_state[config.global_state_mapping["last_agro_action"]] > ModelActions.CALL:
        # check for special case preflop blind situation.
        if global_state[config.global_state_mapping["street"]] == Street.PREFLOP and \
            global_state[config.global_state_mapping["current_player"]] == Positions.BIG_BLIND and \
            global_state[config.global_state_mapping["last_agro_position"]] == Positions.BIG_BLIND:
            action_mask[1] = 1
        else:
            action_mask[0] = 1 # fold is possible
            action_mask[2] = 1 # call is possible
    else:
        # bet
        action_mask[1] = 1 # check is possible
    if max_bet > 0:
        highest_bet_ratio = max_bet / pot
        for i,bet_ratio in enumerate(config.betsizes):
            if bet_ratio <= highest_bet_ratio:
                action_mask[i + 3:] = 1
                break
    return action_mask


def calculate_fixed_limit_mask(global_state,config,action,pot,current_player_investment,current_player_stack):
    """ TODO """
    action_mask = np.zeros(config.num_actions, dtype=int)  # +2 for check and fold
    if global_state[config.global_state_mapping["last_agro_action"]] > ModelActions.CALL:
        max_raise = (pot - current_player_investment) + (2 * global_state[config.global_state_mapping["last_agro_amount"]])
        max_bet = min(current_player_stack,max_raise)
        # check for special case preflop blind situation.
        if global_state[config.global_state_mapping["street"]] == Street.PREFLOP and \
            global_state[config.global_state_mapping["current_player"]] == Positions.BIG_BLIND and \
            global_state[config.global_state_mapping["last_agro_position"]] == Positions.BIG_BLIND:
            action_mask[1] = 1
        else:
            action_mask[0] = 1 # fold is possible
            action_mask[2] = 1 # call is possible
    else:
        # bet
        max_bet = min(current_player_stack,pot)
        action_mask[1] = 1 # check is possible
    if max_bet > 0:
        highest_bet_ratio = current_player_stack / max_bet
        for i,bet_ratio in enumerate(config.betsizes):
            if bet_ratio <= highest_bet_ratio:
                action_mask[i + 3:] = 1
                break
    return action_mask

#### Action Functions ####

def calculate_pot_limit_betsize(last_agro_action,last_agro_amount,config,action,pot,player_street_total,player_stack):
    if last_agro_action > StateActions.CALL:
        # find max raise -> call the raise, raise the pot
        max_raise = (2 * last_agro_amount) + (pot - player_street_total)
        min_raise = max(last_agro_amount + (pot - player_street_total),config.blinds[0])
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(max(max_raise * bet_ratio,min_raise),player_stack + player_street_total)
        return RAISE, bet_amount
    else:
        # bet
        max_bet = pot
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(max_bet * bet_ratio,player_stack)
        return BET, bet_amount
    
def calculate_no_limit_betsize(last_agro_action,last_agro_amount,config,action,pot,player_street_total,player_stack):
    if last_agro_action > StateActions.CALL:
        # find max raise -> call the raise, raise the pot
        min_raise = last_agro_amount + (pot - player_street_total)
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(max(min_raise * bet_ratio,min_raise),player_stack + player_street_total)
        return RAISE, bet_amount
    else:
        # bet
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(pot * bet_ratio,player_stack)
        return BET, bet_amount
    

def calculate_fixed_limit_betsize(last_agro_action,last_agro_amount,config,action,pot,player_street_total,player_stack):
    """ TODO """
    if last_agro_action > StateActions.CALL:
        # find max raise -> call the raise, raise the pot
        max_raise = (2 * last_agro_amount) + (pot - player_street_total)
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(max_raise * bet_ratio,player_stack + player_street_total)
        return RAISE, bet_amount
    else:
        # bet
        max_bet = pot
        bet_ratio = config.betsizes[action - (ModelActions.CALL+1)]
        bet_amount = min(max_bet * bet_ratio,player_stack)
        return BET, bet_amount