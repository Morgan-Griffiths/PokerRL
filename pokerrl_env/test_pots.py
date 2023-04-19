from pokerrl_env.utils import readable_card_to_int
import pytest
import numpy as np
from typing import List, Tuple, Dict
from pokerrl_env.config import Config
from pokerrl_env.datatypes import Player, Street
from pokerrl_env.transition import get_pots, calculate_winnings, game_over

@pytest.fixture
def example_players():
    return [
        Player(1, 100, 1, [8, 14], 5, 100),
        Player(2, 0, 1, [4, 5], 3, 50),
        Player(3, 150, 1, [12, 9], 5, 100)
    ]

@pytest.fixture
def config2():
    return Config(num_players=2)

@pytest.fixture
def config3():
    return Config(num_players=3)

@pytest.fixture
def example_total_amount_invested():
    return {1: 100, 2: 50, 3: 100}

@pytest.fixture
def total_amount_invested_2():
    return {1: 100, 2: 50, 3: 100}


@pytest.fixture
def example_pots(example_players):
    return [
        [150, [example_players[0], example_players[1],example_players[2]]],
        [100, [example_players[0], example_players[2]]]
    ]

def test_simple_pot(example_players):
    pots = get_pots([example_players[0],example_players[2]], {1: 100, 3: 100})
    assert pots == [
        [200, [example_players[0],example_players[2]]]
    ]

def test_get_pots(example_players, example_total_amount_invested,example_pots):
    pots = get_pots(example_players, example_total_amount_invested)
    print('final pots', pots)
    assert pots == example_pots

def test_calculate_winnings(example_pots, example_players):
    winnings = calculate_winnings(example_pots, example_players)
    print('winnings', winnings)
    assert winnings == {
        1: {'hand': [8, 14], 'hand_value': 5, 'result': -50},
        2: {'hand': [4, 5], 'hand_value': 3, 'result': 100},
        3: {'hand': [12, 9], 'hand_value': 5, 'result': -50}
    }

# Add more tests for game_over function if needed

def test_game_over_no_side_pots(config2):
    global_state = np.zeros(config2.global_state_shape)
    # set stacks for two players
    board_cards = [('A', 's'), ('K', 's'), ('Q', 's'), ('J', 's'), ('T', 's')]
    board = [readable_card_to_int(card) for card in board_cards]
    hand1 = [('A', 'h'), ('K', 'h'), ('Q', 'h'), ('J', 'h')]
    hand1 = [readable_card_to_int(card) for card in hand1]
    hand2 = [('A', 'c'), ('K', 'c'), ('9', 's'), ('8', 's')]
    hand2 = [readable_card_to_int(card) for card in hand2]
    # flatten hand1 and hand2
    hand1 = [item for sublist in hand1 for item in sublist]
    hand2 = [item for sublist in hand2 for item in sublist]
    board = [item for sublist in board for item in sublist]
    hands = [hand1,hand2]
    print(board)
    print('hands', hands)
    print(global_state.shape)
    total_amount_invested = {6: 25, 2: 25}
    for i,position in enumerate([2,6]):
        global_state[config2.global_state_mapping[f'player_{position}_stack']] = 100
        global_state[config2.global_state_mapping[f'player_{position}_active']] = 1
        global_state[config2.global_state_mapping[f'player_{position}_position']] = position
        global_state[config2.global_state_mapping[f'player_{position}_hand_range'][0]:config2.global_state_mapping[f'player_{position}_hand_range'][1]] = hands[i]

    global_state[config2.global_state_mapping['board_range'][0]:config2.global_state_mapping['board_range'][1]] = board
    global_state[config2.global_state_mapping['street']] = Street.RIVER
    global_state[config2.global_state_mapping['pot']] = 50
    winnings = game_over(global_state, config2, total_amount_invested)
    print('winnings',winnings)
    assert winnings[2]['result'] == -25
    assert winnings[6]['result'] == 25

def test_game_over_with_side_pots(config3):
    global_state = np.zeros(config3.global_state_shape)
    # set stacks for two players
    board_cards = [('A', 's'), ('K', 's'), ('Q', 's'), ('J', 's'), ('T', 's')]
    board = [readable_card_to_int(card) for card in board_cards]
    hand1 = [('A', 'h'), ('K', 'h'), ('Q', 'h'), ('J', 'h')]
    hand1 = [readable_card_to_int(card) for card in hand1]
    hand2 = [('A', 'c'), ('K', 'c'), ('Q', 'c'), ('J', 'c')]
    hand2 = [readable_card_to_int(card) for card in hand2]
    hand3 = [('A', 'c'), ('K', 'c'), ('9', 's'), ('8', 's')]
    hand3 = [readable_card_to_int(card) for card in hand3]
    # flatten hand1 and hand2
    hand1 = [item for sublist in hand1 for item in sublist]
    hand2 = [item for sublist in hand2 for item in sublist]
    hand3 = [item for sublist in hand3 for item in sublist]
    board = [item for sublist in board for item in sublist]
    hands = [hand1,hand2,hand3]
    stacks = [100, 100,0]
    for i,position in enumerate([1,2,6]):
        global_state[config3.global_state_mapping[f'player_{position}_stack']] = stacks[i]
        global_state[config3.global_state_mapping[f'player_{position}_active']] = 1
        global_state[config3.global_state_mapping[f'player_{position}_position']] = position
        global_state[config3.global_state_mapping[f'player_{position}_hand_range'][0]:config3.global_state_mapping[f'player_{position}_hand_range'][1]] = hands[i]
    total_amount_invested = {1: 50, 2: 50, 6:25}
    global_state[config3.global_state_mapping['board_range'][0]:config3.global_state_mapping['board_range'][1]] = board
    global_state[config3.global_state_mapping['street']] = Street.RIVER
    global_state[config3.global_state_mapping['pot']] = 125

    winnings = game_over(global_state, config3,total_amount_invested)
    print(winnings)
    assert winnings[1]['result'] == -25
    assert winnings[2]['result'] == -25
    assert winnings[6]['result'] == 50
