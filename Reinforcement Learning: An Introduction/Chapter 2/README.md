**Exercise 2.1**

In the comparison shown in Figure 2.1, which method will perform best in the long run in terms of cumulative reward and cumulative probability of selecting the best action? How much better will it be? Express your answer quantitatively

- In the long run, model with smaller epsilon will perform better in both ways. Given enough time, even small amount of exploration will be enough to learn the optimal actions, and all additional exploration will be useless(assuming stationary environment).

- For epsilon = 0.1: Once the algorithm finds the best policy, it will select the optimal action 90% of the time. In
    the other 10%, it will select randomly. When selecting randomly, it will select the optimal
    action 10% of the time. So, 90% + 10%*10% = 91%.

- For epsilon = 0.01: 99% + 1%*10% = 99.1%

---

**Exercise  2.2** 

Give pseudocode for a complete algorithm for the n-armed bandit problem. Use greedy action selection and incremental computation of action values with α=1/k step-size parameter. Assume a function bandit(a) that takes an action and returns a reward. Use arrays and variables; do not subscript anything by the time index t (for examples of  this style of pseudocode, see Figures 4.1 and 4.3). Indicate how the action values are initialized and updated after each reward. Indicate how the step-size parameters are set for each action as a function of how many times it has been tried.

```
Actions are represented by integers from [0,n)
Initalize K as an array of value 0 for all n actions
Initalize Q as an array of value 0 for all n actions
for each step:
  action = argmax(Q)
  K[action] += 1
  reward = bandit(action)
  alpha = 1/K[action]
  Q[action] += alpha * (reward - Q[action])
```
---

**Exercise 2.4**

- See the jupyter notebook

---

**Exercise 2.5**

The results shown in Figure 2.2 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? What might make this method perform particularly better or worse, on average, on particular early plays?

- Since all Q values are optimistic almost always(assuming Gaussian distribution with mean = 0 and variance = 1), the agent first tries each action(in random order) and reduces their Q values. Q value of the best action gets reduced the least. Since agent is choosing greedily, on 11th step, optimal action will be chosen nearly every time. Why is it only ~40% then?

---

**Exercise 2.6**

Suppose you face a binary bandit task whose true action values change randomly from play to play. Specifically, suppose that for any play the true values of actions 1 and 2 are respectively 0.1 and 0.2 with probability 0.5(case A), and 0.9 and 0.8 with probability 0.5 (case B). If you are not able to tell which case you face at any play, what is the best expectation of success you can achieve and how should you behave to achieve it?

- You should expect to achieve 0.5*(0.9+0.1)/2 + 0.5*(0.8+0.2)/2 = 0.5 regardless of how you behave.

Now suppose that on each play you are told if you are facing case A or case B (although you still don’t know the true action values). This is an associative search task. What is the best expectation of success you can achieve in this task, and how should you behave to achieve it?

- You could learn optimal actions(with method described in exercise 2.2 or in some other way) for each case. Then the expecteded reward is 0.5\*0.2 + 0.5\*0.9 = 0.55
