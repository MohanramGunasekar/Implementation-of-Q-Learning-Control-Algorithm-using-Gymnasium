# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent learns the optimal action-value function by interacting with the environment and updates its Q-values to reach the goal state while avoiding the holes.


## Software Requirements
- Python 3.x
- Gymnasium
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab


## Environment Description
The FrozenLake-v1 environment is a 4 × 4 grid-based environment where the agent must reach the goal while avoiding holes.

The environment contains:

- S – Starting state
- F – Frozen and safe state
- H – Hole state
- G – Goal state

The environment is configured with is_slippery=False, so the agent moves exactly in the direction selected.

```text
FFFH
FHSF
FHFF
GFFF
```


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1. Initialize the FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate, discount factor, and epsilon parameters.
4. Reset the environment at the beginning of each episode.
5. Select an action using the epsilon-greedy policy.
6. Perform the selected action and observe the next state and reward.
7. Update the Q-value using the Q-Learning update rule.
8. Continue until the episode terminates or the maximum number of steps is reached.
9. Reduce epsilon after each episode.
10. Calculate the state-value function and learned policy.
11. Display the Q-table, policy, average reward, and learning curve.


## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
epsilon = epsilon_start
episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    total_reward = 0

    for step in range(max_steps_per_episode):

        action = epsilon_greedy_action(
            state,
            epsilon
        )

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        if done:

            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

            total_reward += reward
            break

        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )







```
---

## Output

```text
Final Q-table:
[[0.139 0.008 0.737 0.163]
 [0.444 0.    0.95  0.676]
 [0.938 0.961 0.    0.951]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.06 ]
 [0.    0.    0.    0.   ]
 [0.    0.97  0.951 0.951]
 [0.961 0.96  0.951 0.   ]
 [0.    0.    0.    0.003]
 [0.    0.    0.    0.   ]
 [0.    0.98  0.961 0.961]
 [0.97  0.97  0.96  0.951]
 [0.    0.    0.    0.   ]
 [1.    0.99  0.98  0.   ]
 [0.99  0.98  0.97  0.97 ]
 [0.98  0.97  0.97  0.96 ]]


Estimated State-Value Function:
[[0.737 0.95  0.961 0.   ]
 [0.06  0.    0.97  0.961]
 [0.003 0.    0.98  0.97 ]
 [0.    1.    0.99  0.98 ]]


Learned Policy:
[['R' 'R' 'D' 'L']
 ['U' 'L' 'D' 'L']
 ['U' 'L' 'D' 'L']
 ['L' 'L' 'L' 'L']]



Average reward over last 1000 episodes: 0.967
```

---

## Result
The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the action values through repeated interaction with the environment and developed a policy to reach the goal while avoiding the holes.

## Inference

```text

The experiment shows that Q-Learning can learn an effective policy through repeated interaction with the environment. The epsilon-greedy strategy allows the agent to explore different actions initially and gradually exploit the learned Q-values as epsilon decreases.

```

---

