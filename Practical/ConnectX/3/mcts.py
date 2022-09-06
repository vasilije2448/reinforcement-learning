from kaggle_environments import make, evaluate
import numpy as np
import random
import time
import os
import pickle
from logger import logger, init_logger
from multiprocessing import Process, Queue

class Node:
    def __init__(self, state, parent, player_id, action, game_over, result):
        self.state = state
        self.num_visits = 1
        self.num_wins = 0
        self.parent = parent
        self.children = []
        self.player_id = player_id # indicates who made the _previous_ move
        self.action = action # action taken by parent to get to this node
        self.game_over = game_over
        self.result = result
        
# selection
# expansion
# simulation
# backpropagation
def mcts_process(env, my_player_id, max_time, queue):
    start_time = time.time()
    opponent_player_id = 1 if my_player_id == 2 else 2
    env.state[my_player_id-1].status = 'ACTIVE'
    env.state[opponent_player_id-1].status = 'INACTIVE'
    root = Node(env.state[0], None, opponent_player_id, None, False, 0)
    while (
        time.time() - start_time < max_time
    ):
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
                env2 = make("connectx", state=selected_node.state)
                player_id = 1 if selected_node.player_id == 2 else 2
                result = random_playout(env2, player_id)
        else:
            result = selected_node.result

        if result is None:
            if selected_node.player_id == 1:
                # first player made an invalid move
                result = -1
            else:
                result = 1

        #  Backpropagation
        while selected_node is not None:
            if result == 1:
                if selected_node.player_id == 1:
                    selected_node.num_wins += 1
            if result == -1:
                if selected_node.player_id == 2:
                    selected_node.num_wins += 1
            if result == 0:
                selected_node.num_wins += 0.5
                
            selected_node = selected_node.parent
    best_root_child = max(root.children, key=lambda x: x.num_wins/x.num_visits)
    queue.put(root.children)


# multiprocessing: https://stackoverflow.com/a/45829852
def mcts(env, my_player_id, num_cpu, max_time):
    q = Queue()
    processes = []
    rets = []
    for _ in range(num_cpu):
        p = Process(target=mcts_process, args=(env, my_player_id, max_time, q))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get()
        rets.append(ret)
    for p in processes:
        p.join()
    action_to_num_wins = [0]*7
    action_to_num_visits = [1]*7
    for root_children in rets:
        for ch in root_children:
            action_to_num_wins[ch.action] += ch.num_wins
            action_to_num_visits[ch.action] += ch.num_visits
    action_to_wr = [0]*7
    for i in range(7):
        action_to_wr[i] = action_to_num_wins[i] / action_to_num_visits[i]
    return np.argmax(action_to_wr)

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
    c = 2**0.5
    return node.num_wins/node.num_visits + c*(parent.num_visits**0.5)/node.num_visits

def _expand(node):
    if node.game_over:
        return
    other_player_id = 1 if node.player_id == 2 else 2
    for i in range(7):
        env3 = make("connectx", state=node.state)
        env3.state[0] = node.state
        env3.state[0].observation.mark = 1
        env3.state[other_player_id-1].status = 'ACTIVE' # this is a hack, not sure why it's like this
        env3.state[node.player_id-1].status = 'INACTIVE'
        env3.step([i,i])
        game_over = (env3.state[node.player_id-1].status in ['DONE', 'INVALID'] or env3.state[other_player_id-1] in ['DONE', 'INVALID'])
        result = env3.state[0].reward
        if not game_over:
            env3.state[node.player_id-1].status = 'ACTIVE' # there has to be a better way to do this
            env3.state[other_player_id-1].status = 'INACTIVE'
        child = Node(env3.state[0], node, other_player_id, i, game_over, result)
        node.children.append(child)
        
def random_playout(env, player_id):
    """
    Plays 1 game until the end, random but legal moves only, returns the result
    from first player's perspective.
    """
    env.steps.append(env.steps[0])
    env.state[player_id-1].status = 'ACTIVE'
    env.run(['random', 'random'])
    return env.state[0].reward

