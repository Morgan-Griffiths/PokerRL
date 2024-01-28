# Overview

This is a poker environment for reinforcement learning.

# Installation

run

```
python build.py
```

# Instructions

To instantiate the environment, pass in the config.

the config consists of the following parameters:

- number of players (2-6), default 2
- bet limit (fixed limit, no limit, pot limit), default pot limit
- bet sizes allowed (array of floats), default (1, 0.9, 0.75, 0.67, 0.5, 0.33, 0.25, 0.1)
- Game type [Holdem, OmahaHI], default OmahaHI

## Example usage

This is the recommended way to use the environment.

```
from pokerrl import Config, Game

config = Config(
    num_players=2,
    bet_limit=BetLimits.POT_LIMIT,
    bet_sizes=[1, 0.5],
    game_type=GameTypes.OMAHA_HI,
)
game = Game(config)
player_state,done,winnings,action_mask = game.reset()
while not done:
  action = model(player_state)
  player_state,done,winnings,action_mask = game.step(action)
```

## Play a game (both sides)

```
from pokerrl import play_game
play_game()
```

## Example usage (low level)

```
from pokerrl import Config, init_state, step_state, GameTypes, BetLimits, player_view, Positions, get_current_player

config = Config(
    num_players=2,
    bet_limit=BetLimits.POT_LIMIT,
    bet_sizes=[1, 0.5],
    game_type=GameTypes.OMAHA_HI,
)
global_state,done,winnings,action_mask = init_state(config)
while not done:
  player_idx = get_current_player(global_state)
  player_state = player_view(global_state, player_idx)
  action = model(player_state)
  global_state,done,winnings,action_mask = step_state(global_state, action, config)
```

## Player view (low level)

```
from pokerrl import Config, init_state, step_state, GameTypes, BetLimits, player_view, Positions, get_current_player

config = Config()
global_state,done,winnings,action_mask = init_state(config)
while not done:
  player_idx = get_current_player(global_state)
  human_readable_view(global_state,player_idx, config)
  action = get_action(action_mask,global_state,config)
  global_state,done,winnings,action_mask = step_state(global_state, action, config)
```

## State

The state is an array. To get a player's view of the state, pass the state into the view with the appropriate player index.

## Design decisions

- Record the total amount raised.
  If you record the actual amount raised this means its more difficult to tell what the raise size is when facing multiple raises. But easier to tell what the raise size is when facing a single raise. Also complicates the process of determining how much a player has to call, as the raise size is in relation to the previous bet, which in multiplayer games, is not necessarily the current player.
- Global state player numbers are identical to their position.
- SB and BB posts are the first two states.

- A raise is the total amount. Subtract player street total
- A call vs a raise is the difference = villain bet - player street total
- A call vs a bet is the full amount = villain bet - player street total
- A bet is the full amount
- A fold is 0
