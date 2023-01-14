from OneStepNegamaxQLearning import AgentDQN

a = AgentDQN(policy_net_path = 'OneStepNegamaxQLearning/policy_net3.pt')
a.train(100_000)
