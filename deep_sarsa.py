import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import copy


class DeepSARSA:
    """
    The Deep SARSA algorithm implementation.
    """

    def __init__(self, q_network, policy, env, memory, gamma=0.99, epsilon=0.05, alpha=0.001, batch_size=32):
        self.q_network = q_network
        self.target_q_network = copy.deepcopy(q_network).eval()
        self.policy = policy
        self.env = env
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.batch_size = batch_size
        self.optim = AdamW(q_network.parameters(), lr=alpha)

    def train(self, episodes):
        """
        Train the Deep SARSA algorithm for a specified number of episodes.
        """
        stats = {'MSE Loss': [], 'Returns': []}

        for episode in tqdm(range(1, episodes + 1)):
            state = self.env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.policy(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.q_network(state_b).gather(1, action_b)
                    next_action_b = self.policy(next_state_b, self.epsilon)
                    next_qsa_b = self.target_q_network(next_state_b).gather(1, next_action_b)
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.q_network.zero_grad()
                    loss.backward()
                    self.optim.step()
                    stats['MSE Loss'].append(loss.item())

                state = next_state
                ep_return += reward.item()

            stats['Returns'].append(ep_return)

            if episode % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

        return stats
