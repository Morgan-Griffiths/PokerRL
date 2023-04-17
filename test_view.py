import numpy as np
import pytest
from view import player_view, human_readable_view,return_board_cards
from config import Config
from transition import init_state


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def initial_states(config):
    return init_state(config)


@pytest.fixture
def player_index():
    return 1

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

def test_player_view(initial_states, player_index,config):
    print('test_player_view',player_index)
    global_state,_,_,_ = initial_states
    player_states = player_view(global_state, player_index,config)
    print(player_states[:,:30])
    assert player_states.shape == (2, 50), "Player view should have a shape of (2, 50)."
    assert np.all(player_states[:, 20] == player_index), f"Player index should be consistent in the player view. {player_states[:, 20]}, {player_index}"


def test_human_readable_view(initial_states, player_index,config):
    global_state,_,_,_ = initial_states
    human_readable_states = human_readable_view(global_state, player_index,config)
    for state in human_readable_states:
        assert "hand_range" in state, "Hand range should be present in the human-readable view."
        assert "board_range" in state, "Board range should be present in the human-readable view."
        assert "street" in state, "Street should be present in the human-readable view."
        assert "num_players" in state, "Number of players should be present in the human-readable view."
        assert "hero_position" in state, "Hero position should be present in the human-readable view."

        for key, value in state.items():
            assert value is not None, f"{key} should not be None in the human-readable view. {state}"
