import random
import numpy as np
import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env import ConnectFourGym
from util import kaggle_state_to_train_state, get_win_percentages_kaggle, kaggle_state_to_bitboard
from game import get_valids
from constants import NUM_COLUMNS, LOGDIR, TB_LOGS_DIR, MODEL_DIR, BEST_MODEL_PATH

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

class Agent1:
    def __init__(self, ppo_model):
        self.ppo_model = ppo_model
    def kaggle_agent(self, observation, configuration):
        train_state = kaggle_state_to_train_state(observation.board).to(device)
        action_values = self.ppo_model.policy.evaluate_actions(train_state, tensor([x for x in
                                                                                    range(NUM_COLUMNS)]).to(device))
        Q = tensor([x for x in action_values[1]])
        Q += abs(Q.min()) + 1 # make all values positive
        bitboard = kaggle_state_to_bitboard(observation.board)
        valids = get_valids(bitboard)
        Q *= valids # make illegal actions 0
        action = np.argmax(Q)
        return int(action)
   
class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(Net, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc3 = nn.Linear(128, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        return x

if __name__ == '__main__':
    env = Monitor(ConnectFourGym())
    eval_env = DummyVecEnv([lambda:Monitor(ConnectFourGym(is_eval=True))])

    policy_kwargs = {
        'activation_fn':th.nn.ReLU, 
        'net_arch':[64, dict(pi=[32, 16], vf=[32, 16])],
        'features_extractor_class':Net,
        'normalize_images':False
    }

    model = PPO(policy = 'CnnPolicy', env=env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log =
                TB_LOGS_DIR, batch_size=512, learning_rate=0.001)


    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                                    log_path=LOGDIR, eval_freq=5000,
                                    deterministic=True, render=False,
                                    verbose=1, n_eval_episodes=1000)

    model.set_env(env)
    model.learn(total_timesteps=500_000, callback=eval_callback)
    model = PPO.load(BEST_MODEL_PATH) 
    agent = Agent1(model)

    get_win_percentages_kaggle(agent1=agent.kaggle_agent, agent2="random", n_rounds=1000)
    get_win_percentages_kaggle(agent1=agent.kaggle_agent, agent2="negamax", n_rounds=10)
