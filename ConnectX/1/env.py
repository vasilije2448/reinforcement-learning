import gym
import numpy as np
import random

from constants import NUM_ROWS, NUM_COLUMNS
from game import starting_bitboard, get_valids, play_move, is_game_over, get_legal_moves
from util import bitboard_to_train_state, bitboard_to_kaggle_state

class ConnectFourGym:
    def __init__(self, warmup_timesteps = 100_000, is_eval=False):
        self.bitboard  = starting_bitboard()
        self.current_player_id = 1
        self.action_space = gym.spaces.Discrete(NUM_COLUMNS)
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                            shape=(3, NUM_ROWS, NUM_COLUMNS), dtype=np.float)        
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        
        self.episode_count = 0
        self.timesteps = 0
        self.warmup_timesteps = warmup_timesteps
        self.play_first = True
        self.is_eval = is_eval

    def reset(self):
        self.episode_count += 1
        self.bitboard = starting_bitboard()
        self.current_player_id = 1
        train_state = bitboard_to_train_state(self.bitboard, self.current_player_id)
        return train_state

    def step(self, action):
        self.timesteps += 1
        valids = get_valids(self.bitboard)
        if valids[action]:
            self.bitboard = play_move(self.bitboard, action, self.current_player_id) #TODO
            game_over, winner = is_game_over(self.bitboard, self.current_player_id)
            self.current_player_id = 3 - self.current_player_id
            reward = winner if game_over else 1 / (NUM_ROWS * NUM_COLUMNS)

            # opponent move
            if not game_over:
                new_train_state, opponent_reward, game_over = self.play_random_action()
                if game_over:
                    reward = 1 - opponent_reward
        else:
            reward = -10
            game_over = True
        new_train_state = bitboard_to_train_state(self.bitboard, self.current_player_id)
        return new_train_state, reward, game_over, {}

    # assumes game not over
    def play_random_action(self):
        legal_moves = get_legal_moves(self.bitboard)
        self.bitboard = play_move(self.bitboard, random.choice(legal_moves), self.current_player_id)
        game_over, winner = is_game_over(self.bitboard, self.current_player_id)
        self.current_player_id = 3 - self.current_player_id
        reward = winner if game_over else 1 / (NUM_ROWS * NUM_COLUMNS)
        new_train_state = bitboard_to_train_state(self.bitboard, self.current_player_id)

        return new_train_state, reward, game_over
