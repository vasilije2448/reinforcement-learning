import numpy as np
import random
import time
import os
import torch
from torch.utils.data import TensorDataset, ConcatDataset
from torch.multiprocessing import Process, Queue, set_start_method, Manager
from collections import namedtuple
from tqdm import tqdm

from logger import logger, init_logger
from game import legal_move, get_valids, play_move, is_game_over, starting_bitboard
from neural_net import NetWrapper, Net
from util import kaggle_state_to_bitboard
from constants import NUM_COLUMNS


TrainArgs = namedtuple('TrainArgs', ['num_iterations', 'num_episodes', 'num_mcts_simulations',
                       'epsilon', 'num_cpu'], defaults=(100, 1000, 200, 0.3, 4))
TrainNetArgs = namedtuple('TrainNetArgs', ['num_epochs', 'batch_size', 'learning_rate'],
                          defaults=(20, 1024, 0.01))

CPUCT = 2

class Agent4:
    def __init__(self, net_wrapper, num_mcts_simulations):
        self.net_wrapper = net_wrapper
        self.num_mcts_simulations = num_mcts_simulations

    def kaggle_agent(self, observation, configuration):
        bitboard = kaggle_state_to_bitboard(observation.board)
        action_probabilities = torch.tensor(mcts(bitboard, observation.mark,
                                                  self.num_mcts_simulations, self.net_wrapper))
        valids = get_valids(bitboard)
        action_probabilities *= valids
        return action_probabilities.argmax().item()

def train(train_args, train_net_args, net_wrapper):
    for iteration in tqdm(range(train_args.num_iterations), desc='Iteration'):
        dataset = self_play(train_args, net_wrapper) # generate training data
        torch.save(dataset, f'datasets/dataset{iteration}.pt')
        net_wrapper.train_net(dataset, train_net_args)
        net_wrapper.save_net(f'neural_networks/net{iteration}.pt')

def self_play(train_args, net_wrapper):
    if train_args.num_cpu > train_args.num_episodes:
        raise Exception('More CPUs assigned than episodes to run.')
    q = Manager().Queue()
    processes = []
    rets = []
    episodes_per_process = train_args.num_episodes // train_args.num_cpu
    for _ in range(train_args.num_cpu):
        p = Process(target=run_episodes_process, args=(episodes_per_process,
                                                        train_args.num_mcts_simulations, train_args.epsilon, net_wrapper,
                                                        q))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()
        rets.append(ret)
    for p in processes:
        p.join()

    dataset = ConcatDataset(rets)
    return dataset

def mcts(bitboard, player_id, num_simulations, net_wrapper):
    start_time = time.time()
    opponent_player_id = 3 - player_id
    root = Node(bitboard, None, 0, 0, None, False, player_id, None)
    current_simulation = 0
    while (
        current_simulation < num_simulations
    ):
        current_simulation += 1
        # Selection
        selected_node = select_node(root)
        # Expansion
        V = expand(selected_node, net_wrapper).item()
        # No simulation for now
        # Backpropagation
        while selected_node is not None:
            selected_node.W_v += V
            selected_node.Q = selected_node.W_v / selected_node.N
            V = 1 - V
            selected_node = selected_node.parent

    action_to_num_visits = np.zeros(NUM_COLUMNS)
    for ch in root.children:
        action_to_num_visits[ch.action] = ch.N
    action_probabilities = action_to_num_visits / np.sum(action_to_num_visits)
    return action_probabilities

def run_episodes_process(num_episodes, num_mcts_simulations, epsilon, net_wrapper, queue):
    states_list = []
    action_probabilities_list = []
    values_list = []
    for episode in range(num_episodes):
        bitboard = starting_bitboard()
        game_over = False
        player_id = 1
        episode_experience = []
        result = 0
        while not game_over:
            action_probabilities = torch.tensor(mcts(bitboard, player_id,
                                                      num_mcts_simulations,
                                                      net_wrapper),
                                                dtype=torch.float32)

            valids = get_valids(bitboard)
            action_probabilities *= valids
            s = action_probabilities.sum(0)
            if s == 0:
                game_over = True
                result = 0
                transformed_state = get_state(bitboard, player_id)
                episode_experience.append((transformed_state.cpu(), action_probabilities))
                break
            else: # normalize such that probabilities add up to 1
                action_probabilities = action_probabilities / s

            transformed_state = net_wrapper.get_state(bitboard, player_id)
            flipped_state = flip(transformed_state, 3)
            episode_experience.append((transformed_state.cpu(), action_probabilities))

            exploratory_action = np.random.choice(NUM_COLUMNS, p=action_probabilities.numpy())
            greedy_action = action_probabilities.argmax().item()
            if random.random() < epsilon:
                action_to_play = exploratory_action
            else:
                action_to_play = greedy_action

            bitboard = play_move(bitboard, action_to_play, player_id)
            game_over, result = is_game_over(bitboard, player_id)
            player_id = 3 - player_id
        for e in reversed(episode_experience):
            transformed_state = e[0]
            action_probabilities = e[1]

            states_list.append(transformed_state)
            action_probabilities_list.append(action_probabilities)
            values_list.append(result)

            flipped_state = flip(transformed_state, 3)
            states_list.append(flipped_state)
            flipped_ap = flip(action_probabilities, 0)
            action_probabilities_list.append(flipped_ap)
            values_list.append(result)

            result = 1 - result
    s_tensor = torch.stack(states_list)
    q_tensor = torch.stack(action_probabilities_list)
    v_tensor = torch.tensor(values_list)
    dataset = TensorDataset(s_tensor, v_tensor, q_tensor)
    queue.put(dataset)


def select_node(root):
    selected_node = root
    while selected_node.children:
        selected_node.N += 1
        selected_node.sqrt_N = selected_node.N ** 0.5
        selected_node = find_node_to_explore(selected_node)
    selected_node.N += 1
    selected_node.sqrt_N = selected_node.N ** 0.5
    return selected_node

def find_node_to_explore(parent_node):
    return max(parent_node.children, key=lambda x: uct_value(x, parent_node.sqrt_N))

def uct_value(node, sqrt_parent_N):
    return node.Q + CPUCT * node.P * sqrt_parent_N / (1 + node.N)


def expand(node, net_wrapper):
    if node.game_over:
        return torch.tensor([node.result])
    other_player_id = 3 - node.player_id
    transformed_state = net_wrapper.get_state(node.state, node.player_id)
    with torch.no_grad():
        Pi, V = net_wrapper.predict(transformed_state.squeeze(0))
    Pi, V = Pi.cpu(), V.cpu()
    valids = get_valids(node.state)
    Pi *= valids
    # TODO: check if all Pi values are 0? Does this ever happen?
    Pi /= Pi.sum()
    for i in range(NUM_COLUMNS):
        if valids[i]:
            new_bitboard = play_move(node.state, i, node.player_id)
            game_over, result = is_game_over(new_bitboard, node.player_id)
            if game_over and V != 1:
                V = torch.tensor([result])
            child = Node(new_bitboard, i, Pi[i].item(), V, node, game_over, other_player_id, result)
        else:
            game_over = True
            result = 1 if node.player_id == 1 else 0
            child = Node(node.state, i, 0, V, node, game_over, other_player_id, result)
        node.children.append(child)
    return V

class Node:
    def __init__(self, state, action, P, first_play_urgency, parent, game_over, player_id, result):
        self.state = state
        self.action = action
        self.P = P # prior probability
        self.W_v = 0 # estimate of total action-value, accumulated over N leaf evaluations
        self.N = 0 # number of visits
        self.sqrt_N = 0
        self.Q = first_play_urgency # combined mean action-value
        self.parent = parent
        self.game_over = game_over
        self.player_id = player_id
        self.children = []
        self.result = result # from player_id's perspective, 1 if win, 0 if loss, else 0.5

def flip(t, dim=0):
    """
    inputs:
    t - torch tensor
    dim - dimension to flip (currently only 1 dimension supported)
    outputs:
    t_flipped - input t with dimension dim flipped
    """
    dim_size = t.size()[dim]
    reverse_indices = torch.arange(dim_size-1, -1, -1, device=t.device)
    return t.index_select(dim, reverse_indices)

# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

if __name__ == '__main__':
    set_start_method('spawn')
    train_args = TrainArgs()
    train_net_args = TrainNetArgs()
    net_wrapper = NetWrapper()
    train(train_args, train_net_args, net_wrapper)
