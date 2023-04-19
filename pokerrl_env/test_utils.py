from pokerrl_env.datatypes import BetLimits, GameTypes, Player, Positions
from pokerrl_env.utils import is_next_player_the_aggressor,return_deck
import pytest
from pokerrl_env.config import Config
from pokerrl_env.transition import init_state
import numpy as np


def test_default_values():
    config = Config()
    assert config.game_type == GameTypes.OMAHA_HI
    assert config.num_players == 2
    assert config.bet_limit == BetLimits.POT_LIMIT
    assert config.betsizes == (1, 0.9, 0.75, 0.67, 0.5, 0.33, 0.25, 0.1)
    assert config.blinds == (0.5,1)
    assert config.stack_sizes == 100

def test_custom_values():
    config = Config(game_type=GameTypes.OMAHA_HI, num_players=4, bet_limit=BetLimits.POT_LIMIT,
                    betsizes=(1, 0.75, 0.5, 0.25), blinds=(1, 0.5), stack_sizes=500)
    assert config.game_type == GameTypes.OMAHA_HI
    assert config.num_players == 4
    assert config.bet_limit == BetLimits.POT_LIMIT
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

def test_is_next_player_the_aggressor():
    active_players = [Player(Positions.BIG_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.DEALER, 100, 1, [8, 14], 5, 10),]
    current_player = 2
    last_agro_player = 1
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == False


def test_is_next_player_the_aggressor2():
    active_players = [Player(Positions.BIG_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10)]
    current_player = 2
    last_agro_player = 6
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == True

def test_is_next_player_the_aggressor3():
    active_players = [Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.BIG_BLIND, 100, 1, [8, 14], 5, 10)]
    current_player = 2
    last_agro_player = 6
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == True

def test_is_next_player_the_aggressor4():
    active_players = [Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.BIG_BLIND, 100, 1, [8, 14], 5, 10)]
    current_player = 2
    last_agro_player = 5
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == True


def test_is_next_player_the_aggressor5():
    active_players = [Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.DEALER, 100, 1, [8, 14], 5, 10)]
    current_player = 6
    last_agro_player = 2
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == False


def test_is_next_player_the_aggressor6():
    active_players = [Player(Positions.UTG, 100, 1, [8, 14], 5, 10),Player(Positions.DEALER, 100, 1, [8, 14], 5, 10)]
    current_player = 6
    last_agro_player = 2
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == True


def test_is_next_player_the_aggressor7():
    active_players = [Player(Positions.SMALL_BLIND, 100, 1, [8, 14], 5, 10),Player(Positions.DEALER, 100, 1, [8, 14], 5, 10)]
    current_player = 1
    last_agro_player = 2
    result = is_next_player_the_aggressor(active_players, current_player, last_agro_player)
    assert result == True