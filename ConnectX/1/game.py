import numpy as np
import torch
import random

from constants import NUM_ROWS, NUM_COLUMNS, NUM_IN_A_ROW, TOP_ROW_MASK, FULL_BOARD_MASK

def starting_bitboard():
    """
    Returns: 2 bitboards. First shows whether a square is occupied and second which player occupies
        it(0 if player1, 1 if player2).
    """
    return np.array([0, 0], dtype=np.int64)

# taken from
# https://www.kaggle.com/code/jamesmcguigan/connectx-vectorized-bitshifting-tutorial#get_gameovers()
#@njit(int64[:]())
def get_gameovers():
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

GAMEOVERS = get_gameovers()
LEGAL_MOVES_CACHE = np.array([
    [
        1 << j & i == 0
        for j in range(NUM_COLUMNS - 1, -1, -1)
    ]
    for i in range(2 ** NUM_COLUMNS)
], dtype=np.int8)

def legal_move(occupancy_bitboard, move):
    top_row = occupancy_bitboard & TOP_ROW_MASK
    top_row = top_row >> (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
    return LEGAL_MOVES_CACHE[top_row, move]

TOP_ROW_TO_LEGAL_MOVES = []
for i in range(2 ** NUM_COLUMNS):
    TOP_ROW_TO_LEGAL_MOVES.append([])
    for j in range(NUM_COLUMNS):
        if legal_move(occupancy_bitboard=i<<(NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS), move=j):
            TOP_ROW_TO_LEGAL_MOVES[i].append(j)

TOP_ROW_TO_VALIDS = []
for i in range(2 ** NUM_COLUMNS):
    TOP_ROW_TO_VALIDS.append([])
    for j in range(NUM_COLUMNS):
        if legal_move(occupancy_bitboard=i<<(NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS), move=j):
            TOP_ROW_TO_VALIDS[i].append(1)
        else:
            TOP_ROW_TO_VALIDS[i].append(0)
    TOP_ROW_TO_VALIDS[i] = np.array(TOP_ROW_TO_VALIDS[i], dtype=np.float32)


def get_bitboard(observation):
    """
    Returns 2 bitboards. First shows whether a square is occupied and second which player occupies
    it.
    """
    b1 = 0
    b2 = 0
    for i in range(NUM_ROWS * NUM_COLUMNS): 
        if observation.board[NUM_ROWS * NUM_COLUMNS - 1 - i] == 0:
            continue
        elif observation.board[NUM_ROWS * NUM_COLUMNS - 1 - i] == 1:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
        else:
            current_bit_mask = 1 << i
            b1 |= current_bit_mask
            b2 |= current_bit_mask
    return np.array([b1, b2], dtype=np.int64)

def find_index(bitboard, move):
    move = NUM_COLUMNS - 1 - move
    for row in range(NUM_ROWS):
        current_index = move + row*NUM_COLUMNS
        m = 1 << current_index
        if m & bitboard[0] == 0:
            return current_index
    return 1_000 # should never happen

def play_move(bitboard, move, player_id):
    """
    Assumes that the move is legal
    """
    mark = 0 if player_id == 1 else 1
    index = find_index(bitboard, move)
    return np.array([
        bitboard[0] | 1 << index,
        bitboard[1] | mark << index
    ], dtype=np.int64)


# taken from
# https://www.kaggle.com/code/jamesmcguigan/connectx-vectorized-bitshifting-tutorial#get_winner()
# modified to return result relative to player_id
#@njit
def get_winner(bitboard, player_id):
    """ 
    Endgame get_winner: 0.5 for no get_winner, 1 if player_id won, 0 if other player won
    Q: Why even check if other player won? Player can only win when he makes the move.
    Returns: game_over, winner
    """
    p2_wins = (bitboard[0] &  bitboard[1]) & GAMEOVERS[:] == GAMEOVERS[:]
    if np.any(p2_wins): return (True, 1) if player_id == 2 else (True, 0)
    p1_wins = (bitboard[0] & ~bitboard[1]) & GAMEOVERS[:] == GAMEOVERS[:]
    if np.any(p1_wins): return (True, 1) if player_id == 1 else (True, 0)
    return False, 0.5

def no_possible_moves(bitboard):
    return bitboard[0] == FULL_BOARD_MASK

def is_game_over(bitboard, player_id):
    """ 
    :returns: game_over, result
    result is 0.5 = draw/not over, 1 = player_id won, 0 = other player won
    """
    if no_possible_moves(bitboard):
        return True, 0.5
    else:
        game_over, winner = get_winner(bitboard, player_id)
        return game_over, winner

def get_legal_moves(bitboard):
    top_row_bits = bitboard[0] & TOP_ROW_MASK
    top_row_bits = top_row_bits >> (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
    return TOP_ROW_TO_LEGAL_MOVES[top_row_bits]

def get_valids(bitboard):
    """
    Returns a torch tensor of size NUM_COLUMNS, where value is 1 if the move is legal and 0 otherwise. 
    """
    top_row_bits = bitboard[0] & TOP_ROW_MASK
    top_row_bits = top_row_bits >> (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS)
    return TOP_ROW_TO_VALIDS[top_row_bits]
