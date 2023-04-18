import pytest
from pokerrl.config import Config
from pokerrl.transition import init_state
import numpy as np

from utils import return_deck

def test_default_values():
    config = Config()
    assert config.game_type == 'OmahaHI'
    assert config.num_players == 6
    assert config.bet_limit == 'Pot limit'
    assert config.betsizes == (1, 0.9, 0.75, 0.67, 0.5, 0.33, 0.25, 0.1)
    assert config.blinds == (1, 0.5)
    assert config.stack_sizes == 1000

def test_custom_values():
    config = Config(game_type='OmahaHI', num_players=4, bet_limit='Pot limit',
                    betsizes=(1, 0.75, 0.5, 0.25), blinds=(1, 0.5), stack_sizes=500)
    assert config.game_type == 'OmahaHI'
    assert config.num_players == 4
    assert config.bet_limit == 'Pot limit'
    assert config.betsizes == (1, 0.75, 0.5, 0.25)
    assert config.blinds == (1, 0.5)
    assert config.stack_sizes == 500

def test_init_state():
    config = Config(num_players=4)
    state,done,winnings,action_mask = init_state(config)
    assert state.shape == (2,91)
    assert isinstance(state, np.ndarray)

def test_return_deck():
    deck = return_deck()
    assert len(deck) == 52
    assert isinstance(deck, list)

