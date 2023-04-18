
RAISE = "Raises"
BET = "Bets"
FOLD = "Folds"
CALL = "Calls"
CHECK = "Checks"
BLIND = "blind"

class GameTypes:
    HOLDEM = 'Holdem'
    OMAHA_HI = 'OmahaHi'
    OMAHA_HI_LO = 'OmahaHiLo'

class Positions:
    SMALL_BLIND = 1
    BIG_BLIND = 2
    UTG = 3
    UTG_1 = 4
    UTG_2 = 5
    DEALER = 6

class Player:
    def __init__(self,position,stack,active,hand=None,hand_value=None,total_invested=0):
        self.position = int(position)
        self.stack = stack
        self.active = active
        self.hand = hand
        self.hand_value = hand_value
        self.total_invested = total_invested

    def __repr__(self):
        return f'Position:{self.position}, Stack:{self.stack}, Active:{self.active}, Hand:{self.hand}'

class BetLimits:
    POT_LIMIT = 'Pot limit'
    NO_LIMIT = 'No limit'
    FIXED_LIMIT = 'Fixed limit'

class StateActions:
    PADDED = 0
    FOLD = 1
    CHECK = 2
    CALL = 3

class ModelActions:
    FOLD = 0
    CHECK = 1
    CALL = 2


class Street:
    PADDED = 0
    PREFLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4

INT_TO_STREET = {
    0: '_',
    1: 'PREFLOP',
    2: 'FLOP',
    3: 'TURN',
    4: 'RIVER'
}
BOARD_CARDS_GIVEN_STREET = {
    0: 0,
    1: 0,
    2: 6,
    3: 8,
    4: 10,
}

POSITION_TO_SEAT = {
    None: 0,
    'Small Blind': 1,
    'Big Blind': 2,
    'UTG':3,
    'UTG+1':4,
    'UTG+2':5,
    'Dealer':6
}
SEAT_TO_POSITION = {v: k for k, v in POSITION_TO_SEAT.items()}

rake_cap = {
    "low": {
        2: 0.5,
        3: 1,
        4: 1,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
    },
    "high": {
        2: 1,
        3: 2,
        4: 3,
        5: 3,
        6: 4,
        7: 4,
        8: 4,
        9: 4,
    },
}


ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
suits = ["c", "d", "h", "s"]
# card_to_int = {}
rank_to_int = {rank: i for i, rank in enumerate(ranks, start=1)}
rank_to_int["_"] = 0
suit_to_int = {suit: i for i, suit in enumerate(suits, start=1)}
suit_to_int["_"] = 0
# idx = 1  # zero padded
# for rank in range(0, 13):
#     for suit in range(0, 4):
#         card_to_int[ranks[rank] + suits[suit]] = idx
#         idx += 1
# int_to_card = {v: k for k, v in card_to_int.items()}
int_to_rank = {v: k for k, v in rank_to_int.items()}
int_to_suit = {v: k for k, v in suit_to_int.items()}
int_to_suit[0] = "-"
int_to_rank[0] = "-"
preflop_positions = ["UTG", "UTG+1", "UTG+2", "Dealer", "Small Blind", "Big Blind"]
preflop_order = {POSITION_TO_SEAT[p]: i for i,p in enumerate(preflop_positions,start=1)} # values change preflop to postflop
postflop_positions = [
    "Small Blind",
    "Big Blind",
    "UTG",
    "UTG+1",
    "UTG+2",
    "Dealer",
]
postflop_order = {POSITION_TO_SEAT[p]: POSITION_TO_SEAT[p] for p in postflop_positions}
PREFLOP_ORDER_BY_NUM_PLAYERS = {
    2: ['Dealer', 'Big Blind'],
    3: ['Dealer', 'Small Blind', 'Big Blind'],
    4: ['UTG','Dealer', 'Small Blind', 'Big Blind'],
    5: ['UTG','UTG+1','Dealer', 'Small Blind', 'Big Blind'],
    6: ['UTG','UTG+1','UTG+2','Dealer', 'Small Blind', 'Big Blind'],
}
POSTFLOP_ORDER_BY_NUM_PLAYERS = {
    2: ['Big Blind', 'Dealer'],
    3: ['Small Blind', 'Big Blind', 'Dealer'],
    4: ['Small Blind', 'Big Blind', 'UTG', 'Dealer'],
    5: ['Small Blind', 'Big Blind', 'UTG', 'UTG+1', 'Dealer'],
    6: ['Small Blind', 'Big Blind', 'UTG', 'UTG+1', 'UTG+2', 'Dealer'],
}
PLAYER_ORDER_BY_STREET = {
    Street.PREFLOP: preflop_order,
    Street.FLOP: postflop_order,
    Street.TURN: postflop_order,
    Street.RIVER: postflop_order,
}
PLAYERS_POSITIONS_DICT = {
    2: ["Dealer", "Big Blind"],
    3: ["Small Blind", "Big Blind", "Dealer"],
    4: ["Small Blind", "Big Blind", "UTG", "Dealer"],
    5: ["Small Blind", "Big Blind", "UTG", "UTG+1", "Dealer"],
    6: ["Small Blind", "Big Blind", "UTG", "UTG+1", "UTG+2", "Dealer"],
}
 
INT_POSITIONS_BY_NUM_PLAYERS = {
    2: [6, 2],
    3: [1, 2, 6],
    4: [1, 2, 5, 6],
    5: [1, 2, 4, 5, 6],
    6: [1, 2, 3, 4, 5, 6],
}