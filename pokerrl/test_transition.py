
import pytest
import numpy as np
from pokerrl.transition import *
from pokerrl.config import Config
from pokerrl.datatypes import POSITION_TO_SEAT, SEAT_TO_POSITION, StateActions, Street
from pokerrl.utils import calculate_pot_limit_betsize, readable_card_to_int
from pokerrl.transition import get_action_mask,players_finished,step_state,init_state

config = Config(num_players=6)

@pytest.fixture
def river_state():
    global_state = np.zeros(config.global_state_shape, dtype=np.int32)
    for i in range(1, 4):
        global_state[config.global_state_mapping[f'player_{i}_stack']] = 100
        global_state[config.global_state_mapping[f'player_{i}_active']] = 1
        global_state[config.global_state_mapping[f'player_{i}_position']] = i

    global_state[config.global_state_mapping['last_agro_amount']] = 3
    global_state[config.global_state_mapping['last_agro_action']] = 3
    global_state[config.global_state_mapping['last_agro_position']] = 2
    global_state[config.global_state_mapping['last_agro_bet_is_blind']] = 0
    global_state[config.global_state_mapping['street']] = Street.RIVER
    global_state[config.global_state_mapping['current_player']] = 1
    global_state[config.global_state_mapping['current_player']] = 2
    global_state[config.global_state_mapping['pot']] = 50
    return global_state

def test_clear_last_agro_action():
    global_state = np.zeros(config.global_state_shape)
    global_state[config.global_state_mapping['last_agro_action']] = 1
    global_state[config.global_state_mapping['last_agro_position']] = 1
    global_state[config.global_state_mapping['last_agro_amount']] = 1
    global_state[config.global_state_mapping['last_agro_bet_is_blind']] = 1
    clear_last_agro_action(global_state, config)
    assert global_state[config.global_state_mapping['last_agro_action']] == 0
    assert global_state[config.global_state_mapping['last_agro_position']] == 0
    assert global_state[config.global_state_mapping['last_agro_amount']] == 0
    assert global_state[config.global_state_mapping['last_agro_bet_is_blind']] == 0

def test_players_finished():
    global_state = np.zeros(config.global_state_shape)
    global_state[config.global_state_mapping['player_1_active']] = 1
    global_state[config.global_state_mapping['player_1_stack']] = 1
    global_state[config.global_state_mapping['player_2_active']] = 1
    global_state[config.global_state_mapping['player_2_stack']] = 1
    assert not players_finished(global_state, config)
    global_state[config.global_state_mapping['player_2_active']] = 0
    assert players_finished(global_state, config)
    global_state[config.global_state_mapping['player_2_active']] = 1
    global_state[config.global_state_mapping['player_2_stack']] = 0
    assert players_finished(global_state, config)

def test_order_players_by_street():
    global_state = np.zeros(config.global_state_shape)
    global_state[config.global_state_mapping['street']] = Street.PREFLOP
    for i in range(1, 7):
        global_state[config.global_state_mapping[f'player_{i}_stack']] = 100
        global_state[config.global_state_mapping[f'player_{i}_active']] = 1
        global_state[config.global_state_mapping[f'player_{i}_position']] = i
    active_players = order_players_by_street(global_state, config)
    assert [p.position for p in active_players] == [3, 4, 5, 6, 1, 2]
    assert len(active_players) == 6
    global_state[config.global_state_mapping['street']] = Street.FLOP
    active_players = order_players_by_street(global_state, config)
    assert [p.position for p in active_players] == [1, 2, 3, 4, 5, 6]

def test_increment_players():
    global_state = np.zeros(config.global_state_shape)
    global_state[config.global_state_mapping['street']] = Street.FLOP
    current_player = 1
    for i in range(1, 7):
        global_state[config.global_state_mapping[f'player_{i}_stack']] = 100
        global_state[config.global_state_mapping[f'player_{i}_active']] = 1
        global_state[config.global_state_mapping[f'player_{i}_position']] = i
    global_state[config.global_state_mapping['current_player']] = current_player
    global_state[config.global_state_mapping['next_player']] = 2
    active_players = order_players_by_street(global_state, config)
    print('active_players', active_players)
    increment_players(global_state, active_players, current_player, config)
    assert global_state[config.global_state_mapping['current_player']] == 2
    assert global_state[config.global_state_mapping['next_player']] == 3

def test_new_street_player_order():
    global_state = np.zeros(config.global_state_shape)
    for i in range(1, 7):
        global_state[config.global_state_mapping[f'player_{i}_stack']] = 100
        global_state[config.global_state_mapping[f'player_{i}_active']] = 1
        global_state[config.global_state_mapping[f'player_{i}_position']] = i
    global_state[config.global_state_mapping['street']] = Street.FLOP
    new_street_player_order(global_state, config)
    assert global_state[config.global_state_mapping['current_player']] == 1
    assert global_state[config.global_state_mapping['next_player']] == 2

def test_init_state():
    config = Config(num_players=2)
    global_states,done,winnings,action_mask = init_state(config)
    assert global_states[-1,config.global_state_mapping['last_agro_amount']] == 1
    assert global_states[-1,config.global_state_mapping['player_6_stack']] == 99.5
    assert global_states[-1,config.global_state_mapping['player_2_stack']] == 99
    assert global_states[-1,config.global_state_mapping['last_agro_action']] == 4

def test_step_state():
    global_states,done,winnings,action_mask = init_state(config)
    global_states,done,winnings,action_mask = step_state(global_states, ModelActions.FOLD, config)
    assert global_states.shape == (3,config.global_state_shape)
    assert global_states[2,config.global_state_mapping['street']] == Street.PREFLOP
    assert global_states[2,config.global_state_mapping['previous_amount']] == 0
    assert global_states[2,config.global_state_mapping['previous_action']] == StateActions.FOLD
    assert global_states[2,config.global_state_mapping['previous_position']] == POSITION_TO_SEAT['UTG']

def test_classify_action():
    config = Config()
    action, betsize = classify_action(ModelActions.FOLD, 0, 100, 20,  StateActions.CALL+1, 60, config)
    assert action == FOLD and betsize == 0

    action, betsize = classify_action(ModelActions.CHECK, 0, 100, 20,  StateActions.CALL+1, 60, config)
    assert action == CHECK and betsize == 0

    action, betsize = classify_action(ModelActions.CALL, 0, 100, 20,  StateActions.CALL+1, 60, config)
    assert action == CALL and betsize == 20

    action, betsize = classify_action(ModelActions.CALL+1, 0, 100, 0, StateActions.CHECK, 60, config)
    assert action == BET and betsize == 60

    action, betsize = classify_action(ModelActions.CALL+1, 0, 100, 20, StateActions.CALL+1, 60, config)
    assert action == RAISE and betsize == 100


def test_classify_action_in_game():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    amount_invested = 0.5
    action, betsize = classify_action(ModelActions.FOLD, amount_invested, global_state[-1,config.global_state_mapping[f'player_1_stack']], global_state[-1,config.global_state_mapping['last_agro_amount']],global_state[-1,config.global_state_mapping['last_agro_action']],global_state[-1,config.global_state_mapping['pot']],config)
    assert action == FOLD and betsize == 0

    global_state,done,winnings,action_mask = init_state(config)
    amount_invested = 0.5
    action, betsize = classify_action(ModelActions.CALL, amount_invested, global_state[-1,config.global_state_mapping[f'player_1_stack']], global_state[-1,config.global_state_mapping['last_agro_amount']],global_state[-1,config.global_state_mapping['last_agro_action']],global_state[-1,config.global_state_mapping['pot']],config)
    assert action == CALL and betsize == 0.5

    action, betsize = classify_action(ModelActions.CALL, 0, 100, 20, 4, 60, config)
    assert action == CALL and betsize == 20

    # action, betsize = classify_action(4, 0, 100, 20, 0, 60, config)
    # assert action == BET and betsize == 60

    # action, betsize = classify_action(4, 0, 100, 20, 4, 60, config)
    # assert action == RAISE and betsize == 100


def test_blind_init():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    print(global_state[:,config.global_state_mapping['last_agro_amount']])
    assert global_state[0,config.global_state_mapping['last_agro_amount']] == 0.5
    assert global_state[-1,config.global_state_mapping['last_agro_amount']] == 1
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.DEALER
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND

def test_full_game_checked_to_river():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.DEALER
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.DEALER
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.,3.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.,3.,4.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.,3.,4.,4.]))
    player_amount_invested_per_street, player_total_amount_invested = return_investments(global_state, config)
    assert player_total_amount_invested[2] == 1
    assert player_total_amount_invested[6] == 1
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.,3.,4.,4.,4.]))
    assert global_state[-1,config.global_state_mapping['pot']] == 2
    if winnings[2]['hand_value'] < winnings[6]['hand_value']:
        assert winnings[2]['result'] == 1
        assert winnings[6]['result'] == -1
    elif winnings[2]['hand_value'] > winnings[6]['hand_value']:
        assert winnings[2]['result'] == -1
        assert winnings[6]['result'] == 1
    else:
        assert winnings[2]['result'] == 0
        assert winnings[6]['result'] == 0


def test_full_game_with_raise_to_river():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL + 1, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    print(global_state[:,config.global_state_mapping['street']])
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,2.,2.,3.,3.,4.,4.,4.]))
    assert global_state[-1,config.global_state_mapping['pot']] == 6
    assert done == True


def test_full_game_with_raise_to_allin():
    config = Config(num_players=2,stack_sizes=9)
    global_state,done,winnings,action_mask = init_state(config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL + 1, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL + 1, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL, config)
    print(global_state[:,config.global_state_mapping['street']])
    assert global_state[-1,config.global_state_mapping['pot']] == 18
    assert done == True


def test_full_game_with_raise_to_river_3_players():
    config = Config(num_players=3)
    global_state,done,winnings,action_mask = init_state(config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.DEALER
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.SMALL_BLIND
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.SMALL_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CALL, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.]))
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.SMALL_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.BIG_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.DEALER
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.DEALER
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.SMALL_BLIND
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert global_state[-1,config.global_state_mapping['current_player']] == Positions.SMALL_BLIND
    assert global_state[-1,config.global_state_mapping['next_player']] == Positions.BIG_BLIND
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.,3.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.,3.,4.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.,3.,4.,4.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert done == False
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.,3.,4.,4.,4.]))
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.CHECK, config)
    assert done == True
    print(global_state[:,config.global_state_mapping['street']])
    assert np.array_equal(global_state[:,config.global_state_mapping['street']],np.array([1.,1.,1.,1.,2.,2.,2.,3.,3.,3.,4.,4.,4.,4.]))

### Action Mask


def test_fold_allowed_vs_bet_raise(river_state):
    player_total_amount_invested = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
    action_mask = get_action_mask(river_state, player_total_amount_invested, config)
    print(action_mask)
    assert action_mask[0] == 1, "Fold should be allowed vs bet/raise"


def test_check_allowed(river_state):
    player_total_amount_invested = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
    river_state[config.global_state_mapping['last_agro_amount']] = 0
    river_state[config.global_state_mapping['last_agro_action']] = 0
    river_state[config.global_state_mapping['last_agro_position']] = 0
    action_mask = get_action_mask(river_state, player_total_amount_invested, config)
    assert action_mask[1] == 1, "Check should be allowed when the action is unopened"


def test_call_allowed(river_state):
    player_total_amount_invested = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
    action_mask = get_action_mask(river_state, player_total_amount_invested, config)
    assert action_mask[2] == 1, "Call should be allowed when the last aggressive action is a bet or raise"


def test_bet_sizes_allowed(river_state):
    player_total_amount_invested = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
    action_mask = get_action_mask(river_state, player_total_amount_invested, config)
    expected_action_mask = np.ones(config.num_actions, dtype=int)
    expected_action_mask[1] = 0 # no check
    assert np.array_equal(action_mask, expected_action_mask), "Bet sizes should be allowed based on stack and pot"

def test_preflop_fold():
    config = Config(num_players=2)
    global_state,done,winnings,action_mask = init_state(config)
    global_state,done,winnings,action_mask = step_state(global_state, ModelActions.FOLD, config)
    assert done == True
    assert winnings[6]['result'] == -0.5
    assert winnings[2]['result'] == 0.5
    assert winnings[6]['hand'] == []

### TEST POT CALCS ###

def test_calculate_pot_limit_betsize_2_player_SB():
    config = Config(num_players=2,stack_sizes=9)
    last_agro_action = StateActions.CALL + 1
    last_agro_amount = 1
    action = ModelActions.CALL + 1
    pot = 1.5
    player_street_total = 0.5
    player_stack = 9
    action_str,betsize = calculate_pot_limit_betsize(last_agro_action, last_agro_amount, config, action, pot, player_street_total, player_stack)
    assert action_str == RAISE
    assert betsize == 3

def test_calculate_pot_limit_betsize_2_player_BB():
    config = Config(num_players=2,stack_sizes=9)
    last_agro_action = StateActions.CALL + 1
    last_agro_amount = 3
    action = ModelActions.CALL + 1
    pot = 4
    player_street_total = 1
    player_stack = 9
    action_str,betsize = calculate_pot_limit_betsize(last_agro_action, last_agro_amount, config, action, pot, player_street_total, player_stack)
    assert action_str == RAISE
    assert betsize == 9

def test_calculate_pot_limit_betsize_2_player_SB_3bet():
    config = Config(num_players=2,stack_sizes=27)
    last_agro_action = StateActions.CALL + 1
    last_agro_amount = 9
    action = ModelActions.CALL + 1
    pot = 12
    player_street_total = 3
    player_stack = 27
    action_str,betsize = calculate_pot_limit_betsize(last_agro_action, last_agro_amount, config, action, pot, player_street_total, player_stack)
    assert action_str == RAISE
    assert betsize == 27

def test_calculate_pot_limit_betsize_3_player():
    config = Config(num_players=3,stack_sizes=9)
    last_agro_action = StateActions.CALL + 1
    last_agro_amount = 1
    action = ModelActions.CALL + 1
    pot = 1.5
    player_street_total = 0
    player_stack = 9
    action_str,betsize = calculate_pot_limit_betsize(last_agro_action, last_agro_amount, config, action, pot, player_street_total, player_stack)
    assert action_str == RAISE
    assert betsize == 3.5

def test_calculate_pot_limit_betsize_3_player_SB_3bet():
    config = Config(num_players=3,stack_sizes=20)
    last_agro_action = StateActions.CALL + 1
    last_agro_amount = 3.5
    action = ModelActions.CALL + 1
    pot = 5
    player_street_total = 0.5
    player_stack = 20
    action_str,betsize = calculate_pot_limit_betsize(last_agro_action, last_agro_amount, config, action, pot, player_street_total, player_stack)
    assert action_str == RAISE
    assert betsize == 11.5