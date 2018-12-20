import collections
import gym
import numpy as np
import time

NEW_LINE_FREQUENCY = 100


class BaseAgent:
  """The base class for all agents."""

  def __init__(self,
               env_id='CartPole-v0',
               monitor_class=None,
               max_training_episodes=100_000,
               max_episode_steps=1000,
               render_frequency=500,
               exit_on_reward_threshold=False,
               ):
    self.env = gym.make(env_id)
    self.env.render()
    self.env.env.viewer.window.set_location(
        1204, self.env.env.viewer.window.get_location()[1])
    self.env.render()
    self.env.env.viewer.window.on_close = exit
    self.state_size = self.env.observation_space.shape[0]
    self.num_actions = self.env.action_space.n

    self.monitor = monitor_class(self) if monitor_class is not None else None

    self.max_training_episodes = max_training_episodes
    self.max_episode_steps = min(max_episode_steps,
                                 self.env.spec.max_episode_steps)
    self.render_frequency = render_frequency
    self.exit_on_reward_threshold = exit_on_reward_threshold
    self.reward_threshold = self.env.spec.reward_threshold
    self.reward_threshold_trials = self.env.spec.trials

  def act(self, state):
    raise NotImplemented

  def step(self, state, action, reward, next_state, done):
    raise NotImplemented

  def train_agent(self):
    recent_scores = collections.deque(maxlen=self.reward_threshold_trials)
    for i in range(1, self.max_training_episodes + 1):
      score = self.run_episode(train=True)
      recent_scores.append(score)
      recent_scores_mean = np.mean(recent_scores)
      print(f'\rEpisode: {i} '
            f'\tAvg Score: {recent_scores_mean:.3f}',
            end='\n' if i % NEW_LINE_FREQUENCY == 0 else '')
      if i % self.render_frequency == 0:
        self.demo_agent(loop=False, print_steps=False)
      if (self.exit_on_reward_threshold and
          recent_scores_mean >= self.reward_threshold):
        print(f'The reward threshold of f{self.reward_threshold} has been '
              f'achieved.')
        return

  def run_episode(self, train=False, render=False, print_steps=False):
    score = 0
    state = self.env.reset()
    if render:
      self.env.render()
    for step in range(self.max_episode_steps):
      action = self.act(state)
      next_state, reward, done, _ = self.env.step(action)
      if train:
        self.step(state, action, reward, next_state, done)
      if self.monitor is not None:
        self.monitor.update(self, state, action, reward, next_state, done)
      state = next_state
      score += reward
      if render:
        self.env.render()
        if print_steps:
          state_str = ', '.join(f'{x: .5f}' for x in state)
          print(f'Step: {step + 1:>3}   '
                f'Action: {action}   '
                f'State: {state_str}   '
                f'Reward: {reward: >4g}   '
                f'Score: {score: >4g}')
      if done:
        break
    return score

  def demo_agent(self, loop=True, print_steps=True, delay_between_episodes=2):
    while True:
      score = self.run_episode(render=True, print_steps=print_steps)
      print(f'Episode Score: {score:g}\n')
      if not loop:
        break
      time.sleep(delay_between_episodes)
