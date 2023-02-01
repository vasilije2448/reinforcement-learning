from neural_net import NetWrapper
from game import starting_bitboard, get_legal_moves, get_valids, is_game_over, play_move
from util import (bitboard_to_train_state, kaggle_state_to_bitboard, get_win_percentages_kaggle,
    ReplayMemory, Transition, bitboard_to_kaggle_state)
from constants import BEST_NET_PATH, CURRENT_NET_PATH

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import count
from collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TrainArgs = namedtuple('TrainArgs', ['num_episodes', 'target_update', 'test_vs_negamax',
                                     'memory_size', 'epsilon'],
                       defaults=(100_000, 400, True, 120_000, 0.2))
TrainNetArgs = namedtuple('TrainNetArgs', ['num_epochs', 'batch_size', 'learning_rate',
                                           'min_experience', 'gamma'],
                          defaults=(4, 32, 5e-4, 6400, 0.99))

class Agent2:
    def __init__(self, net_wrapper):
        self.net_wrapper = net_wrapper
    def kaggle_agent(self, observation, configuration):
        bitboard = kaggle_state_to_bitboard(observation.board)
        s = bitboard_to_train_state(bitboard, observation.mark)
        s = s.to(device)
        self.net_wrapper.eval()
        with torch.no_grad():
          Q = self.net_wrapper.predict(s).cpu()
        Q += abs(Q.min()) + 1 # make all values positive
        valids = get_valids(bitboard)
        Q *= valids # make illegal actions 0
        return Q.argmax().item()


def select_action(net_wrapper, state, bitboard, epsilon, deterministic=False):
    if random.random() > epsilon or deterministic:
        state = state.to(device)
        with torch.no_grad():
          Q = net_wrapper.predict(state).cpu()
        Q += abs(Q.min()) + 1 # make all values positive
        valids = get_valids(bitboard)
        Q *= valids # make illegal actions 0
        action = Q.argmax().item()
    else:
        action = select_random_action(bitboard)
    return action

def select_random_action(bitboard):
    legal_moves = get_legal_moves(bitboard)
    return random.choice(legal_moves)

def select_opponent_action(net_wrapper, state, bitboard, epsilon, deterministic=False):
    if random.random() > epsilon or deterministic:
        state = state.to(device)
        with torch.no_grad():
          Q = net_wrapper.predict(state).cpu()
        Q += abs(Q.min()) + 1 # make all values positive
        valids = get_valids(bitboard)
        Q *= valids # make illegal actions 0
        action = Q.argmax().item()
    else:
        action = select_random_action(bitboard)
    return action

def train(train_args, train_net_args, net_wrapper):
    memory = ReplayMemory(train_args.memory_size)
    best_test_result = 0

    for i_episode in tqdm(range(train_args.num_episodes)):
        current_bitboard = starting_bitboard()
        current_player_id = 1
        state = bitboard_to_train_state(current_bitboard, current_player_id)
        for t in count():
            action = select_action(net_wrapper, state, current_bitboard, train_args.epsilon)
            current_bitboard = play_move(current_bitboard, action, current_player_id)
            done, reward = is_game_over(current_bitboard, current_player_id)
            current_player_id = 3 - current_player_id
            reward = torch.tensor([reward])
            action = torch.tensor([action])

            if not done:
                next_state = bitboard_to_train_state(current_bitboard, current_player_id)
            else:
                next_state = None
            memory.push_regular_and_flipped(state, action, next_state, reward)
            state = next_state
            if done:
                break

        if i_episode % train_args.target_update == 0:
            net_wrapper.train_net(memory, train_net_args)
            net_wrapper.save_net(CURRENT_NET_PATH)

        if train_args.test_vs_negamax and i_episode % (train_args.target_update*7) == 0 and i_episode > 0:
            agent = Agent2(net_wrapper)
            print(f"episode {i_episode}")
            print('test vs random')
            get_win_percentages_kaggle(agent.kaggle_agent, 'random', n_rounds=1000)
            print('test vs negamax')
            win_pct = get_win_percentages_kaggle(agent.kaggle_agent, 'negamax', n_rounds=100)

            if win_pct > best_test_result:
                net_wrapper.save_net(BEST_NET_PATH)
                best_test_result = win_pct
            print(f'memory_len: {len(memory)}')
            print(f'best test result (vs negamax): {best_test_result}')

    
if __name__ == '__main__':
    train_args = TrainArgs()
    train_net_args = TrainNetArgs()
    net_wrapper = NetWrapper()
    train(train_args, train_net_args, net_wrapper)
