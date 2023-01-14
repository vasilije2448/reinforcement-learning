from kaggle_environments import make, evaluate
import numpy as np
import random
import time
import os
import pickle

from logger import logger, init_logger
from multiprocessing import Process, Queue
from util import bitboard_to_kaggle_state
from game import (legal_move, get_legal_moves, play_move, is_game_over, starting_bitboard,
    get_bitboard_from_kaggle_obs)
from constants import NUM_COLUMNS

C_UCT = 2 ** 0.5

init_logger(logger)

class Agent3:
    def __init__(self, num_cpu, time_per_move):
        self.num_cpu = num_cpu
        self.time_per_move = time_per_move

    def kaggle_agent(self, observation, configuration):
        return int(_mcts(observation, self.num_cpu, self.time_per_move))

class Node:
    def __init__(self, state, parent, player_id, action, game_over, result):
        self.state = state
        self.num_visits = 1
        self.num_wins = 0
        self.parent = parent
        self.children = []
        self.player_id = player_id # indicates whose turn is it to play
        self.action = action # action taken by parent to get to this node
        self.game_over = game_over
        self.result = result

def _mcts_process(observation, max_time, queue):
    start_time = time.time()
    bitboard = get_bitboard_from_kaggle_obs(observation)
    root = Node(bitboard, None, observation.mark, None, False, 0)
    num_simulations = 0
    while (
        time.time() - start_time < max_time
    ):
        num_simulations += 1
        # Selection
        selected_node = _select_node(root)
        if not selected_node.game_over:
            # Expansion
            _expand(selected_node)
            # Simulation
            selected_node = random.choice(selected_node.children)
            if selected_node.game_over:
                result = selected_node.result
            else:
                bb = np.copy(selected_node.state)
                result = _random_playout(bb, selected_node.player_id)
        else:
            result = selected_node.result

        #  Backpropagation
        while selected_node is not None:
            selected_node.num_wins += result
            result = 1 - result
            selected_node = selected_node.parent

    best_root_child = max(root.children, key=lambda x: x.num_wins/x.num_visits)
    queue.put(root.children)

# multiprocessing: https://stackoverflow.com/a/45829852
def _mcts(observation, num_cpu, max_time):
    q = Queue()
    processes = []
    rets = []
    for _ in range(num_cpu):
        p = Process(target=_mcts_process, args=(observation, max_time, q))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()
        rets.append(ret)
    for p in processes:
        p.join()
    action_to_num_wins = [0] * NUM_COLUMNS
    action_to_num_visits = [1]* NUM_COLUMNS
    for root_children in rets:
        for ch in root_children:
            action_to_num_wins[ch.action] += ch.num_wins
            action_to_num_visits[ch.action] += ch.num_visits
    action_to_wr = [0] * NUM_COLUMNS
    for i in range(NUM_COLUMNS):
        action_to_wr[i] = action_to_num_wins[i] / action_to_num_visits[i]
        logger.info(f'''action {i}\n\twr:{action_to_wr[i]}\n\tnum_visits:{action_to_num_visits[i]}\n\tnum_wins: {action_to_num_wins[i]}''')
    max_action = np.argmax(action_to_wr)
    return max_action

def _select_node(root):
    selected_node = root
    while selected_node.children:
        selected_node.num_visits += 1
        selected_node = _find_node_to_explore(selected_node.children, selected_node)
    selected_node.num_visits += 1   
    return selected_node

def _find_node_to_explore(nodes, parent):
    return max(nodes, key=lambda x: _uct_value(x, parent))

def _uct_value(node, parent):
    return node.num_wins / node.num_visits + C_UCT * (parent.num_visits**0.5) / node.num_visits


def _expand(node):
    other_player_id = 3 - node.player_id
    for i in range(NUM_COLUMNS):
        if legal_move(node.state[0], i):
            new_bitboard = play_move(node.state, i, node.player_id)
            game_over, result = is_game_over(new_bitboard, node.player_id)
            child = Node(new_bitboard, node, other_player_id, i, game_over, result)
        else:
            game_over = True
            result = 1 if node.player_id == 1 else 0
            child = Node(node.state, node, other_player_id, i, game_over, result)
        node.children.append(child)
        
def _random_playout(bitboard, starting_player_id):
    """
    Plays 1 game until the end, random but legal moves only, returns the result
    from non starting player's perspective. This is because opponent always moves first in random
    playout.
    Assumes the game is not over before the initial step.
    """
    game_over = False
    player_id = starting_player_id
    non_starting_player_id = 3 - starting_player_id
    while not game_over:
        legal_moves = get_legal_moves(bitboard)
        random_legal_move = random.choice(legal_moves)
        bitboard = play_move(bitboard, random_legal_move, player_id)
        game_over, result = is_game_over(bitboard, non_starting_player_id)
        player_id = 3 - player_id
    return result
