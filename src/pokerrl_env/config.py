from pokerrl_env.datatypes import CALL, CHECK, FOLD, INT_POSITIONS_BY_NUM_PLAYERS, PREFLOP_ORDER_BY_NUM_PLAYERS, POSITION_TO_SEAT, PLAYERS_POSITIONS_DICT, BetLimits, GameTypes, Street
from pokerrl_env.utils import calculate_fixed_limit_mask, calculate_no_limit_betsize, calculate_pot_limit_betsize, calculate_pot_limit_mask, calculate_no_limit_mask, return_deck

class Config:
    def __init__(self, game_type=GameTypes.OMAHA_HI, num_players=2, bet_limit=BetLimits.POT_LIMIT,
                 betsizes=(1, 0.9, 0.75, 0.67, 0.5, 0.33, 0.25, 0.1),
                 blinds=(0.5,1), stack_sizes=100,is_server=False):
        assert num_players >= 2, "Number of players must be at least 2"
        assert bet_limit in [BetLimits.POT_LIMIT, BetLimits.NO_LIMIT, BetLimits.FIXED_LIMIT], "Bet limit must be one of Pot limit, No limit, or Fixed limit"
        assert len(betsizes) > 0, "Betsizes must be a non-empty tuple"
        assert len(blinds) == 2, "Blinds must be a tuple of length 2"
        assert stack_sizes > 0, "Stack sizes must be a positive integer"
        self.is_server = is_server
        self.game_type = game_type
        self.num_players = num_players
        self.bet_limit = bet_limit
        self.betsizes = betsizes
        self.blinds = blinds
        self.stack_sizes = stack_sizes
        self.action_strs = [FOLD, CHECK, CALL]
        self.action_type_to_int = {a: i for i, a in enumerate(self.action_strs, start=1)}
        self.action_betsize_to_int = {b: i for i, b in enumerate(betsizes, start=4)}
        self.action_to_int = self.action_type_to_int | self.action_betsize_to_int
        self.action_to_int['_'] = 0
        self.action_to_str = {v: k for k, v in self.action_to_int.items()}
        self.action_to_str[0] = "None"
        self.num_actions = len(betsizes) + len(self.action_strs)
        self.player_positions = INT_POSITIONS_BY_NUM_PLAYERS[num_players]
        self.player_state_mapping, self.player_state_shape,self.global_state_mapping,self.global_state_shape = return_mappings(game_type)
        if bet_limit == BetLimits.POT_LIMIT:
            self.return_action_mask = calculate_pot_limit_mask
            self.return_betsize = calculate_pot_limit_betsize
        elif bet_limit == BetLimits.NO_LIMIT:
            self.return_action_mask = calculate_no_limit_mask
            self.return_betsize = calculate_no_limit_betsize
        elif bet_limit == BetLimits.FIXED_LIMIT:
            raise NotImplementedError("Fixed limit not implemented yet")
            # self.return_action_mask = calculate_fixed_limit_mask
            # self.return_betsize = calculate_fixed_limit_betsize
        else:
            raise ValueError("Bet limit must be one of Pot limit, No limit, or Fixed limit")

    @staticmethod
    def convert_model_action_to_state(model_action):
        return model_action + 1
    

def return_mappings(game_type):
    """ In the future will return different mappings depending on the game type """
    if game_type == GameTypes.OMAHA_HI:
        player_state_mapping = {
            "hand_range": [0, 8],
            "board_range": [8, 18],
            "street": 18,
            "num_players": 19,
            "hero_position": 20,
            "hero_active": 21,
            "vil_1_active": 22,
            "vil_2_active": 23,
            "vil_3_active": 24,
            "vil_4_active": 25,
            "vil_5_active": 26,
            "vil_1_position": 27,
            "vil_2_position": 28,
            "vil_3_position": 29,
            "vil_4_position": 30,
            "vil_5_position": 31,
            "last_agro_amount": 32,
            "last_agro_action": 33,
            "last_agro_position": 34,
            "last_agro_bet_is_blind": 35,
            "hero_stack": 36,
            "vil_1_stack": 37,
            "vil_2_stack": 38,
            "vil_3_stack": 39,
            "vil_4_stack": 40,
            "vil_5_stack": 41,
            "pot": 42,
            "amount_to_call": 43,
            "pot_odds": 44,
            "previous_amount": 45,
            "previous_position": 46,
            "previous_action": 47,
            "previous_bet_is_blind": 48,
            "next_player": 49,
            "current_player": 50,
        }
        player_state_shape = 51

        global_state_mapping = {
            "player_1_hand_range": [0, 8],
            "player_2_hand_range": [8, 16],
            "player_3_hand_range": [16, 24],
            "player_4_hand_range": [24, 32],
            "player_5_hand_range": [32, 40],
            "player_6_hand_range": [40, 48],
            "board_range": [48, 58],
            "street": 58,
            "num_players": 59,
            "player_1_position": 60,
            "player_2_position": 61,
            "player_3_position": 62,
            "player_4_position": 63,
            "player_5_position": 64,
            "player_6_position": 65,
            "player_1_stack": 66,
            "player_2_stack": 67,
            "player_3_stack": 68,
            "player_4_stack": 69,
            "player_5_stack": 70,
            "player_6_stack": 71,
            "player_1_active": 72,
            "player_2_active": 73,
            "player_3_active": 74,
            "player_4_active": 75,
            "player_5_active": 76,
            "player_6_active": 77,
            "pot": 78,
            "amount_to_call": 79,
            "pot_odds": 80,
            "previous_amount": 81,
            "previous_position": 82,
            "previous_action": 83,
            "previous_bet_is_blind": 84,
            "last_agro_amount": 85,
            "last_agro_action": 86,
            "last_agro_position": 87,
            "last_agro_bet_is_blind": 88,
            "next_player": 89,
            "current_player": 90,
        }
        global_state_shape = 91
    elif game_type == GameTypes.HOLDEM:
        raise NotImplementedError
    elif game_type == GameTypes.OMAHA_HI_LO:
        raise NotImplementedError
    elif game_type == 'BIG_O':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return player_state_mapping, player_state_shape, global_state_mapping, global_state_shape

