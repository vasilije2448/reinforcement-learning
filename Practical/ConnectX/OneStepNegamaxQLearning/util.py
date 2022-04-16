from collections import namedtuple, deque
import numpy as np
import random
import torch
from kaggle_environments import make, evaluate
from copy import deepcopy
from torch.utils.data import Dataset

# layer 1 for my agent's pieces
# layer 2 for opponent's pieces
# layer 3 for empty squares
def get_state(obs, mark, rows=6, columns=7):
    board = np.array(obs['board']).reshape(1, rows, columns)

    layer1 = board[0].copy()
    for c in range(0, columns):
        for r in range(rows - 1, -1, -1):
            value = layer1[r, c]
            if value == 1:
                layer1[r, c] = 1
            else:
                layer1[r, c] = 0

    layer2 = board[0].copy()
    for c in range(0, columns):
        for r in range(rows - 1, -1, -1):
            value = layer2[r, c]
            if value == 2:
                layer2[r, c] = 1
            else:
                layer2[r, c] = 0

    layer3 = board[0].copy()
    for c in range(0, columns):
        for r in range(rows - 1, -1, -1):
            value = layer3[r, c]
            if value == 0:
                layer3[r, c] = 1
            else:
                layer3[r, c] = 0

    if mark == 1:
        board = np.array([[layer1, layer2, layer3]])
    else:
        board = np.array([[layer2, layer1, layer3]])

    return torch.tensor(board).to(device, dtype=torch.float32)

def get_win_percentages(agent1, agent2, n_rounds=100):
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

    return outcomes.count([1,-1])/len(outcomes)

def tensor_state_to_original_board(state):
    """
    Assumes it's my turn.
    """
    state = state.cpu().detach().numpy()[0]
    my_pieces = state[0]
    enemy_pieces = state[1]
    # but who is player 1 and who is player 2?
    
    # check who has more pieces total. Since it's my turn:
    # if num of pieces is equal: I am player 1
    # if opponent has more pieces: I am player 2
    
    # initially, both my and enemy pieces are represented with 1
    if(np.sum(my_pieces) < np.sum(enemy_pieces)):
        my_pieces *= 2 # changes my mark to 2
    else:
        enemy_pieces *= 2 # changes enemy mark to 2
    
    board = np.zeros((6,7))
    board += my_pieces
    board += enemy_pieces
    
    return board.reshape((1,42))[0].astype(int).tolist()


def visualize_game(episode_experience):
    env = make('connectx', debug=True)
    env.run(['random', 'random'])
    random_step = env.steps[0] # this will be copied and overwritten with episode_experience data at each step
    for idx, sasr in enumerate(episode_experience):
        state = sasr[0]
        agent_reward = sasr[3]
        
        step = deepcopy(random_step)
        step[0]['observation']['board'] = tensor_state_to_original_board(state)
        env.steps.insert(idx, step)
        if idx == len(episode_experience)-1:
            print('Agent\'s reward: ' + str(agent_reward.item()))
    env.steps = env.steps[0:len(episode_experience)]
    env.render(mode="ipython")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push_regular_and_flipped(self, *args):
        self.push_regular(*args)
        self.push_flipped(*args)

    def push_regular(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def push_flipped(self, *args):
        """
        Horizontally flip states and action and push to memory
        """
        state = args[0]
        flipped_state = flip(state, 3)
        action = args[1]
        flipped_action = 7 - 1 - action
        next_state = args[2]
        flipped_next_state = None
        if next_state is not None:
            flipped_next_state = flip(next_state, 3)
        reward = args[3]
        self.memory.append(Transition(flipped_state, flipped_action, flipped_next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path): # save yourself
        torch.save(self.memory, path)
    
    def load(self, path):
        self.memory = torch.load(path)

class EpisodeMemory(object):
    """
    ReplayMemory doesn't preserve episode structure, just transitions.
    This is a deque of lists. Each list is an episode.
    Used for visualize_game
    """

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path): # save yourself
        torch.save(self.memory, path)
    
    def load(self, path):
        self.memory = torch.load(path)

