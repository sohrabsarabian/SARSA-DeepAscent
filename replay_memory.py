import random
import torch


class ReplayMemory:
    """
    The experience replay buffer used in the Deep SARSA algorithm.
    """

    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        """
        Insert a transition into the replay buffer.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        """
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        """
        Check if there are enough samples in the replay buffer for a batch.
        """
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)
