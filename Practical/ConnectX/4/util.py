import numpy as np
import torch
from constants import NUM_ROWS, NUM_COLUMNS
from game import starting_bitboard, is_game_over, play_move
from kaggle_environments import make, evaluate

device = 'cpu'

def bitboard_to_train_state(bitboard, player_id):
    """
    layer 1 for current player's pieces
    layer 2 for other player's pieces
    layer 3 for empty squares
    """
    layer1 = np.zeros((6,7))
    layer2 = np.zeros((6,7))
    layer3 = np.zeros((6,7))

    occupancy = bitboard[0]
    player_bitboard = bitboard[1]

    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            idx = 1 << (NUM_ROWS * NUM_COLUMNS - 1 - i * NUM_COLUMNS - j)
            square_occupied = int(bitboard[0] & idx == idx)
            player2_occupies = int(bitboard[1] & idx == idx)
            player1_occupies = int((not player2_occupies) & square_occupied)
            layer1[i,j] = player1_occupies
            layer2[i,j] = player2_occupies
            layer3[i,j] = square_occupied ^ 1

    if player_id == 1:
        board = np.array([[layer1, layer2, layer3]])
    else:
        board = np.array([[layer2, layer1, layer3]])

    return torch.tensor(board).to(device, dtype=torch.float32)

def bitboard_to_kaggle_state(bitboard):
    """
    bitboard is a numpy array of 2 bitboards(np.int64). First shows whether a square is occupied and
    second which player occupies it.
    kaggle_state is a list with values: 0 if empty, 1 if p1, 2 if p2
    """
    ks = [0] * NUM_ROWS * NUM_COLUMNS
    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            bitboard_idx = 1 << (NUM_ROWS * NUM_COLUMNS - 1 - i * NUM_COLUMNS - j)
            list_idx = i * NUM_COLUMNS + j
            square_occupied = int(bitboard[0] & bitboard_idx == bitboard_idx)
            if not square_occupied:
                continue
            player2_occupies = int(bitboard[1] & bitboard_idx == bitboard_idx)
            if player2_occupies:
                ks[list_idx] = 2
            else:
                ks[list_idx] = 1
    return ks

def train_state_to_kaggle_state(train_state):
    """
    Train state is the 3x6x7 state from get_state
    Kaggle state is a list, with values
    0 - if the square is empty
    1 - if the square is occupied by player 1
    2 - if the square is occupied by player 2
    """
    # layer_1 always represents the current player,
    # but need to determine whether that's player1 or player2
    l1_sum = train_state[0].sum().item()
    l2_sum = train_state[1].sum().item()
    player_1_layer = 0 if l1_sum == l2_sum else 1
    player_2_layer = 1 - player_1_layer
    kaggle_state = [0] * NUM_ROWS * NUM_COLUMNS
    for i in range(NUM_ROWS):
        for j in range(NUM_COLUMNS):
            if train_state[player_1_layer][i,j] == 1:
                idx = i * NUM_COLUMNS + j
                kaggle_state[idx] = 1
            elif train_state[player_2_layer][i,j] == 1:
                idx = i * NUM_COLUMNS + j
                kaggle_state[idx] = 2
    return kaggle_state

def kaggle_state_to_bitboard(kaggle_board):
    """
    kaggle_board is list with values: 0 if empty, 1 if p1, 2 if p2
    Returns 2 bitboards. First shows whether a square is occupied and second which player occupies
    it.
    """
    b1 = 0
    b2 = 0
    for i in range(NUM_ROWS * NUM_COLUMNS): 
        if kaggle_board[NUM_ROWS * NUM_COLUMNS - 1 - i] == 0:
            continue
        elif kaggle_board[NUM_ROWS * NUM_COLUMNS - 1 - i] == 1:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
        else:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
            b2 |= current_bit_mask
    return np.array([b1, b2], dtype=np.int64)

def kaggle_state_to_train_state(kaggle_state):
    """
    Train state is the 1x3x6x7 tensor used for training
    layer 1 for current player's pieces
    layer 2 for other player's pieces
    layer 1 for empty squares

    Kaggle state is a list, with values
    0 - if the square is empty
    1 - if the square is occupied by player 1
    2 - if the square is occupied by player 2
    """

    board = np.array(kaggle_state).reshape(NUM_ROWS, NUM_COLUMNS)
    if kaggle_state.count(1) == kaggle_state.count(2):
        current_player = 1
    else:
        current_player = 2

    layer1 = board.copy()
    for c in range(0, NUM_COLUMNS):
        for r in range(NUM_ROWS - 1, -1, -1):
            value = layer1[r, c]
            if value == 1:
                layer1[r, c] = 1
            else:
                layer1[r, c] = 0

    layer2 = board.copy()
    for c in range(0, NUM_COLUMNS):
        for r in range(NUM_ROWS - 1, -1, -1):
            value = layer2[r, c]
            if value == 2:
                layer2[r, c] = 1
            else:
                layer2[r, c] = 0

    layer3 = board.copy()
    for c in range(0, NUM_COLUMNS):
        for r in range(NUM_ROWS - 1, -1, -1):
            value = layer3[r, c]
            if value == 0:
                layer3[r, c] = 1
            else:
                layer3[r, c] = 0

    if current_player == 1:
        board = np.array([[layer1, layer2, layer3]])
    else:
        board = np.array([[layer2, layer1, layer3]])

    return torch.tensor(board).to(device, dtype=torch.float32)

def get_win_percentages_kaggle(agent1, agent2, n_rounds=10):
    """
    Returns agent1's win percentage
    """
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 4))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 4))
    print("Percentage of Invalid Plays by Agent 1:", int(outcomes.count([None, 0])/n_rounds*100))
    print("Percentage of Invalid Plays by Agent 2:", int(outcomes.count([0, None])/n_rounds*100))
