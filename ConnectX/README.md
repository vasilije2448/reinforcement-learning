# ConnectX

https://www.kaggle.com/c/connectx

## Solutions 

#### 1. SB3 Proximal Policy Optimization vs random

Learns to beat the random agent but not much more than that. This was my first attempt at RL and I quickly
realized that Stable Baselines gets in the way more than it helps, at least for multiplayer games.

#### 2. One Step Negamax Q Learning

Learns via self-play. Unlike [standard DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) where expected state action values are calculated as:

$r + \gamma * Q^*(s', a')$

here the second term is negated because in the next state it's the opponent's turn:

$r - \gamma * Q^*(s', a')$

Given that it doesn't use search at all, I'm surprised by how well this works.

#### 3. Monte Carlo Tree Search

Vanilla [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) with root parallelization
(creates multiple independent trees and combines the results at the end).

#### 4. Alpha Zero

More details inside dir 4.
