import os

LOGDIR = os.path.join(".", "logs")
TB_LOGS_DIR = os.path.join(LOGDIR, "tensorboard_logs")
MODEL_DIR = os.path.join(".","models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")

# game related
NUM_ROWS = 6
NUM_COLUMNS = 7
NUM_IN_A_ROW = 4 # need to connect in order to win

# for bitboards
TOP_ROW_MASK = 127 << (NUM_ROWS * NUM_COLUMNS - NUM_COLUMNS) # 111111100000000000000000000000000000000000
FULL_BOARD_MASK = 4398046511103 # 111111111111111111111111111111111111111111
