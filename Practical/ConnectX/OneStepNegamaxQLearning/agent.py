
from OneStepNegamaxQLearning.neural_net import DQN
from OneStepNegamaxQLearning.env import EnvWrapper
from OneStepNegamaxQLearning.util import ReplayMemory, EpisodeMemory, Transition, get_win_percentages
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import count
from kaggle_environments import make, evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgentDQN:
    def __init__(self, policy_net_path='OneStepNegamaxQLearning/policy_net.pt', learning_rate=5e-4, batch_size=32, gamma=0.99, memory_size=120_000,
                 min_experience=6400, num_epochs=4):
        self.policy_net_path = policy_net_path
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.lr = learning_rate
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_size)
        self.episodes_done = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.env_wrapper = EnvWrapper()
        self.min_experience = min_experience
        self.best_test_result = 0
        self.num_epochs = num_epochs

    def select_action(self, state, obs, deterministic=False):
        if random.random() > 0.2 or deterministic:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            r = random.choice([col for col in range(7) if obs.board[int(col)] == 0])
            return torch.tensor([[r]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.min_experience:
            return
        total_loss = 0
        dset = list(self.memory.memory)
        for i in range(self.num_epochs):
            random.shuffle(dset)
            for j in range(int(len(dset)/self.batch_size)):
                transitions = dset[j*self.batch_size:(j+1)*self.batch_size]
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                      batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(self.batch_size, device=device)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

                # Minus because in the next state it's opponent's turn
                expected_state_action_values = -(next_state_values * self.gamma) + reward_batch
                
                criterion = nn.MSELoss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                '''
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                '''
                self.optimizer.step()
        print(f'avg loss: {total_loss/self.num_epochs/(int(len(self.memory)/self.batch_size))}')

    
    def train(self, num_episodes=30_000, target_update=400, test_vs_random=True):
        play_first = True
        for i_episode in tqdm(range(num_episodes)):
            self.episodes_done += 1
            self.env_wrapper.reset()

            if play_first:
                agent_id = 0
                opponent_id = 1
            else:
                agent_id = 1
                opponent_id = 0

                opponent_state = self.get_state(self.env_wrapper.get_current_obs(), mark=opponent_id+1)
                opponent_action = self.select_opponent_action(opponent_state, self.env_wrapper.get_current_obs())

                self.env_wrapper.step(opponent_action.item(), opponent_id)
            state = self.get_state(self.env_wrapper.get_current_obs(), mark=agent_id+1)


            for t in count():
                action = self.select_action(state, self.env_wrapper.get_current_obs())
                reward, done = self.env_wrapper.step(action.item(), agent_id)
                agent_reward = torch.tensor([reward], device=device)

                if not done:
                    opponent_state = self.get_state(self.env_wrapper.get_current_obs(), mark=opponent_id+1)
                    self.memory.push_regular_and_flipped(state, action, opponent_state, agent_reward)

                    opponent_action = self.select_opponent_action(opponent_state,
                                                         self.env_wrapper.get_current_obs())

                    _, done = self.env_wrapper.step(opponent_action.item(), opponent_id)

                    if not done:
                        next_state = self.get_state(self.env_wrapper.get_current_obs(), mark=agent_id+1)

                    else:
                        next_state = None

                else:
                    self.memory.push_regular_and_flipped(state, action, None, agent_reward)


                state = next_state

                if done:
                    play_first = not play_first
                    break

            if i_episode % target_update == 0:
                self.optimize_model()
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_net.eval()
                torch.save(self.policy_net.state_dict(), self.policy_net_path)

            if test_vs_random == True and i_episode % (target_update*7) == 0 and i_episode >= 8400:
                print(f"episode {i_episode}")
                print('test vs random')
                get_win_percentages(self.kaggle_agent, 'random', n_rounds=100)
                print('test vs negamax')
                win_pct = get_win_percentages(self.kaggle_agent, 'negamax', n_rounds=100)

                if win_pct > self.best_test_result:
                    torch.save(self.policy_net.state_dict(), self.policy_net_path)
                    self.best_test_result = win_pct
                print(f'memory_len: {len(self.memory)}')
                print(f'best test result(vs negamax): {self.best_test_result}')

    def select_opponent_action(self, state, obs, deterministic=False):
        if random.random() > 0.2 or deterministic:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            r = random.choice([col for col in range(7) if obs.board[int(col)] == 0])
            return torch.tensor([[r]], device=device, dtype=torch.long)
     
    # layer 1 for my agent's pieces
    # layer 2 for opponent's pieces
    # layer 3 for empty squares
    def get_state(self, obs, mark, rows=6, columns=7):
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
                    #break
                else:
                    layer3[r, c] = 0

        if mark == 1:
            board = np.array([[layer1, layer2, layer3]])
        else:
            board = np.array([[layer2, layer1, layer3]])

        return torch.tensor(board).to(device, dtype=torch.float32)

    def kaggle_agent(self, obs, config): # agent that matches kaggle specification, i.e. works with env and not env_wrapper   
        state = self.get_state(obs, obs.mark, config.rows, config.columns)
        action = self.select_action(state, obs, deterministic=True)
        return int(action)
    
    def kaggle_agent_opponent(self, obs, config):
        state = self.get_state(obs, obs.mark, config.rows, config.columns)
        action = self.select_opponent_action(state, obs, deterministic=True)
        return int(action)

    def load_policy_net(self, policy_net_path, train=False):
        self.policy_net = DQN()
        self.policy_net.load_state_dict(torch.load(policy_net_path))
        self.policy_net.to(device)
        if train:
            self.policy_net.train()
        else:
            self.policy_net.eval()
    
    def policy(self, state):
        with torch.no_grad():
            return self.policy_net(state)      

