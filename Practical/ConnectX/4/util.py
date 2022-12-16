import numpy as np
from constants import NUM_ROWS, NUM_COLUMNS

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
