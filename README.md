# Poker environment for reinforcement learning

To install the environment, pass in the config.

the config consists of the following parameters:

- number of players
- bet limit
- bet sizes allowed
- Game type [Holdem, OmahaHI]

## Views

pass the global state into the view to get the view of the current player

## Actions

integer representating 1 of the available actions.

## State

The state is an array. To get a player's view of the state, pass the state into the view with the appropriate player index.

## Design decisions

- Record the total amount raised.
  If you record the actual amount raised this means its more difficult to tell what the raise size is when facing multiple raises. But easier to tell what the raise size is when facing a single raise. Also complicates the process of determining how much a player has to call, as the raise size is in relation to the previous bet, which in multiplayer games, is not necessarily us.
- Global state player numbers are identical to their position.
- SB and BB posts are the first two states.
