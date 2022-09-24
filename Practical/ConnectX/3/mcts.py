from kaggle_environments import make, evaluate
import numpy as np
import random
import time
import os
import pickle
from logger import logger, init_logger
from multiprocessing import Process, Queue
import unittest

NUM_ROWS = 6
NUM_COLUMNS = 7
NUM_IN_A_ROW = 4 # need to connect in order to win
TOP_ROW_MASK = 127 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS) # 111111100000000000000000000000000000000000
FULL_BOARD_MASK = 4398046511103 # 111111111111111111111111111111111111111111

init_logger(logger)

# taken from
# https://www.kaggle.com/code/jamesmcguigan/connectx-vectorized-bitshifting-tutorial#get_gameovers()
#@njit(int64[:]())
def _get_gameovers():
    """Creates a list of all winning board positions, over 4 directions: horizontal, vertical and 2 diagonals"""
    rows    = NUM_ROWS
    columns = NUM_COLUMNS
    inarow  = NUM_IN_A_ROW

    gameovers = []

    mask_horizontal  = 0
    mask_vertical    = 0
    mask_diagonal_dl = 0
    mask_diagonal_ul = 0
    for n in range(inarow):  # use prange() with numba(parallel=True)
        mask_horizontal  |= 1 << n
        mask_vertical    |= 1 << n * columns
        mask_diagonal_dl |= 1 << n * columns + n
        mask_diagonal_ul |= 1 << n * columns + (inarow - 1 - n)

    row_inner = rows    - inarow
    col_inner = columns - inarow
    for row in range(rows):         # use prange() with numba(parallel=True)
        for col in range(columns):  # use prange() with numba(parallel=True)
            offset = col + row * columns
            if col <= col_inner:
                gameovers.append( mask_horizontal << offset )
            if row <= row_inner:
                gameovers.append( mask_vertical << offset )
            if col <= col_inner and row <= row_inner:
                gameovers.append( mask_diagonal_dl << offset )
                gameovers.append( mask_diagonal_ul << offset )

    return gameovers


GAMEOVERS = _get_gameovers() # masks of every possible win position
LEGAL_MOVES_CACHE = np.array([
    [
        1 << (6-j) & i == 0
        for j in range(NUM_COLUMNS)
    ]
    for i in range(2**NUM_COLUMNS)
], dtype=np.int8)

def _legal_move(occupancy_bitboard, move):
    top_row = occupancy_bitboard & TOP_ROW_MASK
    top_row = top_row >> (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
    return LEGAL_MOVES_CACHE[top_row, move]

TOP_ROW_TO_LEGAL_MOVES = []
for i in range(2**NUM_COLUMNS):
    TOP_ROW_TO_LEGAL_MOVES.append([])
    for j in range(NUM_COLUMNS):
        if _legal_move(occupancy_bitboard=i<<(42-7), move=j):
            TOP_ROW_TO_LEGAL_MOVES[i].append(j)


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
        

def _get_bitboard(observation):
    """
    :returns: 2 bitboards. First shows whether a square is occupied and second which player occupies
        it.
    """
    b1 = 0
    b2 = 0
    for i in range(NUM_ROWS * NUM_COLUMNS): 
        if observation.board[-i] == 0:
            continue
        elif observation.board[-i] == 1:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
        else:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
            b2 |= current_bit_mask
    return np.array([b1, b2], dtype=np.int64)

# selection
# expansion
# simulation
# backpropagation
def mcts_process(observation, max_time, queue):
    start_time = time.time()
    opponent_player_id = 1 if observation.mark == 2 else 2
    bitboard = _get_bitboard(observation)
    root = Node(bitboard, None, opponent_player_id, None, False, 0)
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
                starting_player_id = 1 if selected_node.player_id == 2 else 2
                result = _random_playout(bb, starting_player_id)
        else:
            result = selected_node.result

        ''' think this is no longer needed
        if result is None:
            if selected_node.player_id == 1:
                # first player made an invalid move
                result = -1
            else:
                result = 1
        '''

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

def mcts_test(observation, max_time): # without multiprocessing
    start_time = time.time()
    opponent_player_id = 1 if observation.mark == 2 else 2
    bitboard = _get_bitboard(observation)
    root = Node(bitboard, None, opponent_player_id, None, False, 0)
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
                starting_player_id = 1 if selected_node.player_id == 2 else 2
                result = _random_playout(bb, starting_player_id)
        else:
            result = selected_node.result

        ''' think this is no longer needed
        if result is None:
            if selected_node.player_id == 1:
                # first player made an invalid move
                result = -1
            else:
                result = 1
        '''

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
    for ch in root.children:
        logger.info(f'action: {ch.action}, num_wins: {ch.num_wins}, num_visits: {ch.num_visits}')
    logger.info(f'num simulations: {num_simulations}')
    if best_root_child.action == 6:
        return 0
    else:
        return best_root_child.action + 1
    return best_root_child.action



# multiprocessing: https://stackoverflow.com/a/45829852
def mcts(observation, num_cpu, max_time):
    q = Queue()
    processes = []
    rets = []
    for _ in range(num_cpu):
        p = Process(target=mcts_process, args=(observation, max_time, q))
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
    max_action = np.argmax(action_to_wr)
    if max_action == 6:
        max_action = 0
    else:
        max_action += 1
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
    c = 2**0.5
    return node.num_wins/node.num_visits + c*(parent.num_visits**0.5)/node.num_visits


def _find_index(bitboard, move):
    move = 6 - move
    for row in range(NUM_ROWS):
        current_index = move + row*NUM_COLUMNS
        m = 1 << current_index
        if m & bitboard[0] == 0:
            return current_index
    return 1_000 # should never happen

def _play_move(bitboard, move, player_id):
    """
    Assumes that the move is legal
    """
    mark = 0 if player_id == 1 else 1
    index = _find_index(bitboard, move)
    return np.array([
        bitboard[0] | 1 << index,
        bitboard[1] | mark << index
    ], dtype=np.int64)


# taken from
# https://www.kaggle.com/code/jamesmcguigan/connectx-vectorized-bitshifting-tutorial#get_winner()
#@njit
def _get_winner(bitboard):
    """ Endgame get_winner: 0 for no get_winner, 1 = player 1, 2 = player 2"""
    p2_wins = (bitboard[0] &  bitboard[1]) & GAMEOVERS[:] == GAMEOVERS[:]
    if np.any(p2_wins): return 2
    p1_wins = (bitboard[0] & ~bitboard[1]) & GAMEOVERS[:] == GAMEOVERS[:]
    if np.any(p1_wins): return 1
    return 0

def _no_possible_moves(bitboard):
    return bitboard[0] == FULL_BOARD_MASK

def _game_over(bitboard):
    """
    :returns: game_over, result
    """
    if _no_possible_moves(bitboard):
        return True, 0
    else:
        winner = _get_winner(bitboard)
        return bool(winner), winner # is converting to bool necessary?

def _expand(node):
    if node.game_over: # already checked before?
        return
    other_player_id = 1 if node.player_id == 2 else 2
    for i in range(7):
        if _legal_move(node.state[0], i):
            new_bitboard = _play_move(node.state, i, other_player_id)
            game_over, winner = _game_over(new_bitboard)
            if winner == 0:
                result = 0
            elif winner == 1:
                result = 1
            else:
                result = -1
            child = Node(new_bitboard, node, other_player_id, i, game_over, result)
        else:
            game_over = True
            result = -1 if other_player_id == 1 else 1
            child = Node(node.state, node, other_player_id, i, game_over, result)
        node.children.append(child)
        
def _legal_moves(bitboard):
    top_row_bits = bitboard[0] & TOP_ROW_MASK
    top_row_bits = top_row_bits >> (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
    return TOP_ROW_TO_LEGAL_MOVES[top_row_bits]

def _random_playout(bitboard, starting_player_id):
    """
    Plays 1 game until the end, random but legal moves only, returns the result
    from first player's perspective.
    Assumes the game is not over before the initial step.
    """
    game_over = False
    player_id = starting_player_id
    winner = 0
    while not game_over:
        legal_moves = _legal_moves(bitboard)
        random_legal_move = random.choice(legal_moves)
        bitboard = _play_move(bitboard, random_legal_move, player_id)
        game_over, winner = _game_over(bitboard)
        player_id = 3 - player_id
    if winner == 1:
        return 1
    elif winner == 2:
        return -1
    else:
        return 0


class TestBitboardFunctions(unittest.TestCase):
    def test_legal_moves_1(self):
        bitboard = np.array([0,0], dtype=np.int64)
        lm = _legal_moves(bitboard)
        self.assertEqual(lm, [0, 1, 2, 3, 4, 5, 6])

    def test_legal_moves_2(self):
        bitboard = np.array([0,0], dtype=np.int64)
        bitboard[0] |= 1 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
        lm = _legal_moves(bitboard)
        self.assertEqual(lm, [0, 1, 2, 3, 4, 5])

    def test_legal_moves_3(self):
        bitboard = np.array([0,0], dtype=np.int64)
        bitboard[0] |= 100 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
        lm = _legal_moves(bitboard)
        self.assertEqual(lm, [2, 3, 5, 6])

    def test_legal_move_1(self):
        bitboard = np.array([0,0], dtype=np.int64)
        bitboard[0] |= 100 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
        occupancy = bitboard[0]
        self.assertTrue(_legal_move(occupancy, 2))
        self.assertFalse(_legal_move(occupancy, 4))
        self.assertTrue(_legal_move(occupancy, 6))

    def test_find_index_1(self):
        bitboard = np.array([0,0], dtype=np.int64)
        bitboard[0] = 770525102079
        self.assertEqual(_find_index(bitboard, 6), 21)
        self.assertEqual(_find_index(bitboard, 5), 1000)
        self.assertEqual(_find_index(bitboard, 4), 1000)
        self.assertEqual(_find_index(bitboard, 3), 24)
        self.assertEqual(_find_index(bitboard, 2), 1000)
        self.assertEqual(_find_index(bitboard, 1), 40)
        self.assertEqual(_find_index(bitboard, 0), 27)

    def test_find_index_2(self):
        bitboard = np.array([0,0], dtype=np.int64)
        bitboard[0] = 127
        self.assertEqual(_find_index(bitboard, 0), 13)


    def test_play_move(self):
        bitboard = np.array([0,0], dtype=np.int64)
        new_bitboard = _play_move(bitboard, 0, 1)
        self.assertEqual(new_bitboard[0], 64)
        new_bitboard2 = _play_move(new_bitboard, 0, 1)
        self.assertEqual(new_bitboard2[0], 8256)


if __name__ == '__main__':
    unittest.main()
