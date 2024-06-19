import gym
import torch
from env_utils import PreprocessEnv, plot_stats, plot_cost_to_go, plot_max_q, test_agent, seed_everything
from models import QNetwork
from replay_memory import ReplayMemory
from deep_sarsa import DeepSARSA

# Create and prepare the environment
env = gym.make('MountainCar-v0')
seed_everything(env)

state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

print(f"MountainCar: State dimensions: {state_dims}")
print(f"MountainCar: Number of actions: {num_actions}")

env = PreprocessEnv(env)

state = env.reset()
action = torch.tensor(0)
next_state, reward, done, _ = env.step(action)
print(f"Sample state: {state}")
print(f"Next state: {next_state}, Reward: {reward}, Done: {done}")

# Create the Q-network
q_network = QNetwork(state_dims, num_actions)


# Define the epsilon-greedy policy
def policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)


# Create the replay memory
memory = ReplayMemory()

# Create the Deep SARSA agent
deep_sarsa_agent = DeepSARSA(q_network, policy, env, memory, epsilon=0.01)

# Train the Deep SARSA agent
output_dir = 'output'
stats = deep_sarsa_agent.train(episodes=2500)

# Plot execution stats and save the figure
plot_stats(stats, output_dir)

# Plot the cost-to-go and save the figure
plot_cost_to_go(env, q_network, xlabel='Car Position', ylabel='Velocity', output_dir=output_dir)

# Show the resulting policy and save the figure
plot_max_q(env, q_network, xlabel='Car Position', ylabel='Velocity',
           action_labels=['Back', 'Do nothing', 'Forward'], output_dir=output_dir)

# Test the resulting agent and save the video
test_agent(env, policy, episodes=2, output_dir=output_dir)
