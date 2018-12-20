import collections
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import agents

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQN1StepAgent(agents.BaseAgent):
  def __init__(self,
               fc1_size=64,
               fc2_size=64,
               memory_size=100_000,
               batch_size=64,
               lr=0.0005,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay=0.995,
               gamma=0.99,
               tau=0.001,
               train_frequency=4,
               **kwargs):
    super().__init__(**kwargs)

    self.model = nn.Sequential(
        nn.Linear(self.state_size, fc1_size),
        nn.ReLU(),
        nn.Linear(fc1_size, fc2_size),
        nn.ReLU(),
        nn.Linear(fc2_size, self.num_actions),
    )
    self.target_model = copy.deepcopy(self.model)
    for param in self.target_model.parameters():
      param.requires_grad = False

    self.replay_buffer = ReplayBuffer(memory_size, batch_size)
    self.batch_size = batch_size

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    self.epsilon = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_decay = epsilon_decay
    self.gamma = gamma
    self.tau = tau
    self.train_frequency = train_frequency
    self.step_counter = 0

  def act(self, state):
    if np.random.rand() < self.epsilon:
      # Act randomly.
      action = int(random.random() * self.num_actions)
    else:
      # Act greedily.
      self.model.eval()
      inputs = torch.from_numpy(state).float().unsqueeze(0)
      with torch.no_grad():
        action = self.model(inputs).argmax().item()

    return action

  def step(self, state, action, reward, next_state, done):
    self.replay_buffer.add_sample(state, action, reward, next_state, done)
    self.epsilon = max(self.epsilon_end,
                       self.epsilon * self.epsilon_decay)
    self.step_counter += 1
    if (self.step_counter >= self.train_frequency and
        len(self.replay_buffer) >= self.batch_size):
      self._train()
      self.step_counter = 0

  def _train(self):
    self.model.train()
    state, action, reward, next_state, done = self.replay_buffer.get_batch()

    max_q = self.target_model(next_state).max(1)[0].unsqueeze(1)
    targets = reward + (self.gamma * max_q * (1 - done))
    outputs = self.model(state).gather(1, action)
    loss = F.mse_loss(outputs, targets)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self._soft_update()

  def _soft_update(self):
    for param, target_param in zip(self.model.parameters(),
                                   self.target_model.parameters()):
      target_param.data.copy_(self.tau * param.data +
                              (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
  def __init__(self, memory_size, batch_size):
    self.batch_size = batch_size
    self.memory = collections.deque(maxlen=memory_size)
    self.sample = collections.namedtuple(
        'Sample', ('state', 'action', 'reward', 'next_state', 'done'))

  def add_sample(self, state, action, reward, next_state, done):
    self.memory.append(
        self.sample(state, action, reward, next_state, int(done)))

  def get_batch(self):
    xs = random.sample(self.memory, k=self.batch_size)

    state = torch.from_numpy(np.vstack([x.state for x in xs])).float()
    action = torch.from_numpy(np.vstack([x.action for x in xs])).long()
    reward = torch.from_numpy(np.vstack([x.reward for x in xs])).float()
    next_state = torch.from_numpy(np.vstack([x.next_state for x in xs])).float()
    done = torch.from_numpy(np.vstack([x.done for x in xs])).float()

    return state, action, reward, next_state, done

  def __len__(self):
    return len(self.memory)


def main():
  # config = {
  #   'agent_file': __file__,
  #   # 'env_id': 'CartPole-v0',
  #   # 'env_id': 'Acrobot-v1',
  #   # 'env_id': 'Pendulum-v0',
  #   # 'env_id': 'MountainCar-v0',
  #   # 'env_id': 'LunarLander-v2',
  #   'env_id': 'InvertedDoublePendulum-v2',
  #   'max_episodes': 100_000,
  #   'max_episode_length': 500,
  #   'num_recent_episodes': 100,
  #
  #   # Hyper parameters.
  #   'batch_size': 64,
  #   'epsilon_start': 1.0,
  #   'epsilon_end': 0.1,
  #   'epsilon_decay': 0.995,
  #   'fc1_size': 64,
  #   'fc2_size': 64,
  #   'gamma': 0.99,
  #   'lr': 0.0005,
  #   'memory_size': 100_000,
  #   'tau': 0.001,
  #   'train_frequency': 4,
  # }
  # trainer = Trainer(config)
  # trainer.train_agent()
  pass


if __name__ == '__main__':
  main()
