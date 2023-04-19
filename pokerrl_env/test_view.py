import numpy as np
from pokerrl_env.datatypes import Positions
import pytest
from pokerrl_env.view import player_view, human_readable_view,return_board_cards
from pokerrl_env.config import Config
from pokerrl_env.transition import init_state


@pytest.fixture
def config():
    return Config(num_players=6)


@pytest.fixture
def initial_states(config):
    return init_state(config)


@pytest.fixture
def player_index():
    return 1

@pytest.fixture
def dealer_position():
    return Positions.DEALER

@pytest.fixture
def utg_position():
    return Positions.UTG

def test_return_board_cards(initial_states,config:Config):
    global_state,_,_,_ = initial_states
    board_cards = return_board_cards(global_state[-1],config)
    print(board_cards)
    assert np.array_equal(board_cards,np.zeros(10))
    state = global_state[-1]
    state[config.global_state_mapping['street']] = 2
    board_cards = return_board_cards(state,config)
    assert np.array_equal(board_cards[6:],np.zeros(4))
    state[config.global_state_mapping['street']] = 3
    board_cards = return_board_cards(state,config)
    assert np.array_equal(board_cards[8:],np.zeros(2))
    state[config.global_state_mapping['street']] = 4
    board_cards = return_board_cards(state,config)
    # the next line asserts board_cards has positive numbers only
    assert np.all(board_cards)

def test_player_view(initial_states,utg_position,config:Config):
    print('test_player_view',utg_position)
    global_state,_,_,_ = initial_states
    player_states = player_view(global_state, utg_position,config)
    print(player_states[:,:30])
    assert player_states.shape == (2, config.player_state_shape), "Player view should have a shape of (2,config.player_state_shape)."
    assert np.all(player_states[:, 20] == utg_position), f"Player index should be consistent in the player view. {player_states[:, 20]}, {utg_position}"


def test_human_readable_view(initial_states,config):
    dealer_position = Positions.DEALER
    global_state,_,_,_ = initial_states
    human_readable_states = human_readable_view(global_state, dealer_position,config)
    for state in human_readable_states:
        assert "hand_range" in state, "Hand range should be present in the human-readable view."
        assert "board_range" in state, "Board range should be present in the human-readable view."
        assert "street" in state, "Street should be present in the human-readable view."
        assert "num_players" in state, "Number of players should be present in the human-readable view."
        assert "hero_position" in state, "Hero position should be present in the human-readable view."

