import os

NEURAL_NETS_DIR = os.path.join(".", "neural_networks")
BEST_NET_PATH = os.path.join(NEURAL_NETS_DIR, "best_net.pt")
CURRENT_NET_PATH = os.path.join(NEURAL_NETS_DIR, "current_net.pt")

# game related
NUM_ROWS = 6
NUM_COLUMNS = 7
NUM_IN_A_ROW = 4 # need to connect in order to win

# for bitboards
TOP_ROW_MASK = 127 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS) # 111111100000000000000000000000000000000000
FULL_BOARD_MASK = 4398046511103 # 111111111111111111111111111111111111111111
