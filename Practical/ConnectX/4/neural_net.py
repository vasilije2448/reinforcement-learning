import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from constants import NUM_ROWS, NUM_COLUMNS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(1344, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            #nn.Sigmoid()
            nn.Softmax(1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(1344, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x =  self.cnn_stack(x)
        x = x.flatten(start_dim = 1)
        pi = self.policy_head(x)
        v = self.value_head(x)
        return pi, v

class NetWrapper():
    def __init__(self):
        self.net = Net().to(device)
        self.net.eval()

    def train_net(self, dataset, train_net_args):
        self.net.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=train_net_args.learning_rate)
        dataloader = DataLoader(dataset, batch_size=train_net_args.batch_size)
        print('Training neural net')
        for epoch in range(train_net_args.num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                states, v_labels, pi_labels = batch
                states = torch.squeeze(states)
                states = states.to(device)
                pi_labels = pi_labels.to(device)
                v_labels = v_labels.type(torch.Tensor).to(device)
                optimizer.zero_grad()

                pi, v = self.net(states)
                v = torch.squeeze(v)

                loss_pi = criterion(pi, pi_labels)
                loss_v = criterion(v, v_labels)
                loss = loss_pi + loss_v
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'epoch_loss: {epoch_loss/len(dataloader)}')
        self.net.eval()

    def save_net(self, path):
        torch.save(self.net.state_dict(), path)

    def load_net(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def get_state(self, bitboard, player_id):
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
                layer3[i,j] = square_occupied

        if player_id == 1:
            board = np.array([[layer1, layer2, layer3]])
        else:
            board = np.array([[layer2, layer1, layer3]])

        return torch.tensor(board).to(device, dtype=torch.float32)

    def get_state_kaggle(self, kaggle_observation):
        return None

    def predict(self, state):
        pi, v = self.net.forward(state.unsqueeze(0))
        return pi[0], v[0]

