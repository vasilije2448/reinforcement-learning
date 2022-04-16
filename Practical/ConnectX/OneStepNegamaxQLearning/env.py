from kaggle_environments import make

class EnvWrapper:
    def __init__(self, obs=None):
        self.env  = make("connectx", debug=False)
        if obs != None: # copy other obs' board
            self.env.state[0].observation.board = obs.board[:]
            self.obs = self.env.state[0].observation

        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns   

    def reset(self):
        self.env.reset()

    def step(self, action, player_id):
        is_valid = (self.env.state[0].observation['board'][action] == 0)
        if is_valid:
            self.env.step([action, action]) # only 1 is used, depending on whose turn it is
            self.obs = self.env.state[0].observation
            reward = self.env.state[player_id].reward
            done = self.env.done
        else:
            reward, done, _ = -10., True, {}
        return reward, done
    
    def get_current_obs(self):
        return self.env.state[0].observation
