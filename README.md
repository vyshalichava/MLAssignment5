The objective of this project is to implement and compare two distinct reinforcement learning (RL) algorithms—Q-Learning and Proximal Policy Optimization (PPO)—on a chosen environment from OpenAI Gym. The comparison is made by evaluating the total rewards accumulated by the agents over several episodes. The agents should learn how to maximize the reward by interacting with the environment and adjusting their strategies based on feedback.

The solution involves the following steps:

Implement a Q-Learning agent for the FrozenLake-v1 environment.
Implement a PPO agent for an Atari game environment.
Train both agents, tune their hyperparameters, and evaluate their performance based on cumulative rewards.
Compare the results by visualizing the average rewards for both agents over time.


**Q-Learning Algorithm**
**Overview:**
Q-Learning is a model-free RL algorithm that learns the optimal policy for an agent by updating a Q-value table based on state-action pairs. The agent interacts with the environment, selects actions based on an exploration-exploitation tradeoff, and updates its Q-values according to the rewards received.

Key Components:
Q-Table: A table where each row represents a state, and each column represents an action. Each value in the table (Q-value) represents the expected future reward for taking that action in the given state.
Bellman Equation: Used to update the Q-values:

<img width="444" alt="image" src="https://github.com/user-attachments/assets/e0e5e84c-843c-4740-951d-261a82a1714c">


Where:
α is the learning rate,
γ is the discount factor,
𝑟  is the immediate reward,
s' is the next state.

**Implementation Details:**
Environment: The Q-Learning agent is trained in the FrozenLake-v1 environment, a grid-world problem where the agent must navigate from start to goal without falling into holes.
Hyperparameters:
Learning Rate (α): Controls how much new information overrides old information.
Gamma (γ): The discount factor, which determines the importance of future rewards.
Epsilon (ε): Controls the exploration-exploitation trade-off. Higher values encourage exploration, while lower values favor exploitation.

**Agent's Workflow:**
Initialize Q-values to zero.
At each time step, the agent chooses an action using an ε-greedy policy.
The agent receives a reward and updates its Q-values based on the Bellman equation.
The agent learns by repeating this process over many episodes.

**Training and Tuning:**
The Q-Learning agent is trained with varying combinations of hyperparameters (α, γ, ε) to find the optimal settings:

Learning rates tested: [0.5, 0.8, 0.1]
Gamma values tested: [0.9, 0.95, 0.99]
Epsilon values tested: [0.1, 0.2, 0.3]
For each combination of hyperparameters, the agent's average reward is computed over multiple episodes, and the results are stored in a DataFrame for analysis.

**Results:**
The best combination of hyperparameters yields the highest average reward. The Q-Learning algorithm efficiently learns a policy that maximizes rewards by iterating over many episodes, demonstrating how the agent adapts its strategy based on feedback from the environment.


**Proximal Policy Optimization (PPO):
Overview:**
PPO is an advanced policy-gradient algorithm designed to optimize the agent's policy while maintaining stability during training. It improves the policy in small steps to avoid drastic changes, ensuring a stable learning process.

**Key Components:**
1. Policy Network: The neural network used to select actions based on the current state.
2. Value Network: Estimates the value of a given state, helping the agent learn the long-term benefits of actions.
3. Surrogate Loss Function: PPO maximizes a clipped objective to prevent overly large updates to the policy, which could destabilize learning.
<img width="480" alt="image" src="https://github.com/user-attachments/assets/5d13a03a-e743-468d-9884-898d586ff4ce">

where rt(𝜃) is the probability ratio of new to old policies, and A`t is the advantage estimate.

**Implementation Details:**
Environment: PPO is applied to an Atari game from the OpenAI Gym library. Atari games provide a high-dimensional, visually complex environment where the agent must learn through pixel inputs and delayed rewards.

**Neural Network Architecture:**
Convolutional Layers: Extract features from the visual input (frames from the game).
Fully Connected Layers: Map the extracted features to actions.
Training Process:
The agent collects data by interacting with the environment for several steps.
The policy and value networks are updated using the surrogate loss function and advantage estimates.
PPO uses mini-batches and multiple updates per iteration to improve the policy.
Training and Evaluation:
Hyperparameters:
1. Clip Parameter: Limits the size of policy updates to ensure stability.
2. Gamma: Discount factor to weigh future rewards.
3. GAE Lambda: Controls the trade-off between bias and variance in the advantage estimate.
The training process involves multiple episodes, where the agent collects experience and updates its policy based on the rewards received.

**Results:**
The PPO agent's performance is monitored by logging total rewards over each episode. The agent gradually improves its policy, learning to maximize the cumulative reward over time.

**Output:**
<img width="459" alt="image" src="https://github.com/user-attachments/assets/f5f87148-e373-4a4e-937d-beac11d72771">

**Comparison: Q-Learning vs. PPO**
**Performance Metrics:**
Both algorithms are compared using mean rewards over time, averaged across episodes.
For Q-Learning, the performance is analyzed by testing different hyperparameter combinations, and the results are stored in a DataFrame for comparison.
For PPO, total rewards across episodes are logged and compared to the Q-Learning results.

**Visualization:**
The mean reward for both agents is plotted over time, showing how the Q-Learning agent performs in a simple environment (FrozenLake-v1) compared to the more sophisticated PPO agent, which tackles a more complex Atari game.
The plot provides insights into the learning curves of both agents, showing how quickly they learn and how stable their performance becomes over time.
**Conclusion:**
Q-Learning: This algorithm is effective for simple, discrete environments like FrozenLake, where the state and action space are small enough to be represented by a Q-table. By tuning hyperparameters, the Q-Learning agent can achieve a high average reward, making it an ideal choice for small-scale problems.
PPO: In complex environments like Atari games, where the state space is continuous and high-dimensional, PPO outperforms Q-Learning by leveraging neural networks to approximate policies. PPO's stable updates and ability to handle large state spaces make it a powerful algorithm for visually rich, complex tasks.

**Final Insights:**
Q-Learning: Simple, interpretable, and effective for small environments.
PPO: Sophisticated, scalable, and better suited for high-dimensional environments.
By comparing the results of these two algorithms, it is evident that while Q-Learning is simpler and more interpretable, PPO offers superior performance for larger, more complex tasks.
