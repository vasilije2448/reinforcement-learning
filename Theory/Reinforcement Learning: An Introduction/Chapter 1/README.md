**Exercise  1.1:   Self-Play** 



Suppose, instead of playing against a  random opponent,the reinforcement learning algorithm described above played against itself. What do you think would happen in this case?  Would it learn a different way of playing?



- Yes. Random opponent is usually bad, so the agent might not get to explore a large % of the state space. Self play should fix this.



---



**Exercise 1.2: Symmetries** 



Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the reinforcement learning algorithm described above to take advantage of this?



- We could compress states such that only useful information is preserved.



In what ways would this improve it?



- Training time would be reduced, because the agent wouldn't have to visit as many states.



Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we?



- It depends on our model of the opponent. If the opponent is playing differently in symmetrical positions, we should treat them as different positions. In other words, our state representation should take into account our model of the opponent.



Is it true, then, that symmetrically equivalent positions should necessarily have the same value?



- Not necessarily. It depends on what we're trying to learn. If our goal is to learn how to play well, regardless of the opponent, then we should not take into account what our opponent is doing when representing states. However, if we're trying to learn how to exploit weaker opponents, then we should take into account our model of the particular opponent we're playing against.


---



**Exercise 1.3: Greedy Play** 



Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best.  Would it learn to play better, or worse, than a nongreedy player? What problems might occur?



- It would play worse, because it would only explore a small % of the state space.



---



**Exercise 1.4: Learning from Exploration**



Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time, then the state  values would converge to a set of probabilities. What are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves?



- Exploration doesn't have a negative impact on learning. Learning from exploration too would result in a better policy. This question is confusing.



Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?



- The one that updates values after all moves.



---

