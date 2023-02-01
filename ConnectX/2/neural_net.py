from util import Transition
from constants import NUM_COLUMNS

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1344, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_COLUMNS),
        )

    def forward(self, x):
        x =  self.cnn_stack(x)
        x = x.flatten(start_dim = 1)
        output = self.linear_relu_stack(x)
        return output

class NetWrapper:
    def __init__(self):
        self.policy_net = DQN().to(device)
        self.policy_net.eval()
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def train_net(self, memory, train_net_args):
        if len(memory) < train_net_args.min_experience:
            return
        self.policy_net.train()
        optimizer = optim.Adam(params=self.policy_net.parameters(), lr=train_net_args.learning_rate)
        criterion = nn.MSELoss()
        total_loss = 0
        dset = list(memory.memory)
        bs = train_net_args.batch_size
        for i in range(train_net_args.num_epochs):
            random.shuffle(dset)
            for j in range(int(len(dset) / bs)):
                transitions = dset[j*bs : (j+1)*bs]
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                      batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None]).to(device)
                state_batch = torch.cat(batch.state).to(device)
                action_batch = torch.cat(batch.action).to(device).unsqueeze(1)
                reward_batch = torch.cat(batch.reward).to(device)
                state_action_values = self.policy_net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(bs, device=device)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

                # Minus because in the next state it's opponent's turn
                expected_state_action_values = -(next_state_values * train_net_args.gamma) + reward_batch
                
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                '''
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                '''
                optimizer.step()
        print(f'avg loss: {total_loss/train_net_args.num_epochs/(int(len(memory)/bs))}')
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def save_net(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
        self.target_net.load_state_dict(torch.load(path))
        self.target_net.eval()


    def eval(self):
        self.policy_net.eval()

    def train(self):
        self.policy_net.train()

    def predict(self, state):
        return self.policy_net.forward(state)[0]

