from kaggle_environments import make
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from kaggle_environments import make, evaluate
from gym import spaces
import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from typing import Callable
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

LOGDIR = os.path.join(".","ppo_cnn_vs_random_logs")
MONITOR_LOGS_DIR = os.path.join(LOGDIR,"monitor_logs")
TB_LOGS_DIR = os.path.join(LOGDIR,"tensorboard_logs")
MODEL_DIR = os.path.join(".","models")
CHECKPOINTS_DIR = os.path.join(LOGDIR,"checkpoints")

# divides board into 3 channels - https://www.kaggle.com/c/connectx/discussion/168246
# first channel: player 1 pieces
# second channel: player 2 pieces
# third channel: possible moves. 1 for player_1 and -1 for player_2
def transform_board(board, mark):
    rows = board[0].shape[0]
    columns = board[0].shape[1]

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
                if (mark == 1):
                    layer3[r, c] = 1
                else:
                    layer3[r, c] = -1
                break
            else:
                layer3[r, c] = 0

    board = np.array([[layer1, layer2, layer3]])
    return board

class ConnectFourGym:
    def __init__(self, agent2="random"):
        self.env  = make("connectx", debug=True)
        self.trainer = self.env.train([None, agent2])
        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(3,self.rows,self.columns), dtype=np.float)        
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        
        self.episode_count = 0
        self.timesteps = 0
        self.play_first = True
        self.agent2 = agent2

    def reset(self):
        self.episode_count += 1
        self.obs = self.trainer.reset()
        board_2d = np.array(self.obs['board']).reshape(1,self.rows,self.columns)
        board_3c = transform_board(board_2d, self.obs.mark)
        self.change_trainer()
        return board_3c
    def change_reward(self, old_reward, done):
        if old_reward == 1: # The agent won the game
            return 1
        elif done: # The opponent won the game
            return -1
        else: # Reward 1/42
            return 1/(self.rows*self.columns)
    def step(self, action):
        self.timesteps += 1
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid: # Play the move
            self.obs, old_reward, done, _ = self.trainer.step(int(action))
            reward = self.change_reward(old_reward, done)
        else: # End the game and penalize agent
            reward, done, _ = -10, True, {}
        board_2d = np.array(self.obs['board']).reshape(1,self.rows,self.columns)
        board_3c = transform_board(board_2d, self.obs.mark)
        return board_3c, reward, done, _

    def change_trainer(self):
        if self.play_first:
            self.trainer = self.env.train([None, self.agent2])
        else:
            self.trainer = self.env.train([self.agent2, None])
        self.play_first = not self.play_first

class ModifiedEvalCallback(EvalCallback):
    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "ppo_cnn_vs_random"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True



    
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

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

env = Monitor(ConnectFourGym())
eval_env = DummyVecEnv([lambda:Monitor(ConnectFourGym())])

modified_eval_callback = ModifiedEvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                                log_path=LOGDIR, eval_freq=1000,
                                deterministic=True, render=False,
                                verbose=1, n_eval_episodes=100)

policy_kwargs = {
    'activation_fn':th.nn.ReLU, 
    'net_arch':[64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class':Net,
    'normalize_images':False
}

model = PPO(policy = 'MlpPolicy', env=env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log =
            TB_LOGS_DIR, n_epochs=5, batch_size=128)


start_time = time.time()
model.learn(total_timesteps=350_000, callback=[modified_eval_callback])
end_time = time.time()
print('training took ' + str((end_time - start_time)/3600) + ' hours')

model = PPO.load(os.path.join(MODEL_DIR, "ppo_cnn_vs_random")) 
def agent1(obs, config):
    board_2d = np.array(obs['board']).reshape(1,6,7)
    board_3c = transform_board(board_2d, obs.mark)
    col, _ = model.predict(board_3c, deterministic=True)
    return int(col)

get_win_percentages(agent1=agent1, agent2="random", n_rounds=1000)
