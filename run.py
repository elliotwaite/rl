import argparse
import collections
import importlib
import itertools
import time

import gym
import numpy as np

from utils import mpi

ARGS_OVERRIDES = dict(
    # Atari
    # env='Breakout-v0',
    # env='Enduro-v0',
    # env='MontezumaRevenge-v0',
    # env='MsPacman-v0',
    # env='Pong-v0',
    # env='Qbert-v0',
    # env='Riverraid-v0',
    # env='Seaquest-v0',
    # env='SpaceInvaders-v0',
    # env='VideoPinball-v0',

    # Box2D
    # env='BipedalWalker-v2',
    # env='BipedalWalkerHardcore-v2',
    # env='CarRacing-v0',
    # env='LunarLander-v2',
    # env='LunarLanderContinuous-v2',

    # Classic Control
    env='Acrobot-v1',
    # env='CartPole-v1',
    # env='MountainCar-v0',
    # env='MountainCarContinuous-v0',
    # env='Pendulum-v0',

    alg='keyboard',
    demo=True,
    fps=24,
    num_procs=1
)


class Runner:
  def __init__(self, env, agent):
    self.env = env
    self.agent = agent
    self.renderer_is_initialized = False

  def initialize_renderer(self):
    # Make the close button on the renderer's window exit the program.
    self.env.env.viewer.window.on_close = exit

    # If the renderer's window is resizable, resize it to be as large as
    # possible while preserving it's aspect ration, and center its position.
    if self.env.env.viewer.window.resizeable:
      screen_width = self.env.env.viewer.window.screen.width
      screen_height = self.env.env.viewer.window.screen.height
      aspect_ratio = self.env.env.viewer.window.width / self.env.env.viewer.window.height
      new_width = int(screen_height * aspect_ratio)
      new_height = screen_height
      if new_width > screen_width:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
      self.env.env.viewer.window.set_location(int((screen_width - new_width) / 2),
                                              int((screen_height - new_height) / 2))
      self.env.env.viewer.window.width = new_width
      self.env.env.viewer.window.height = new_height

      self.renderer_is_initialized = True

  def run_episode(self, max_steps=None, train=False, render=False, print_steps=False, fps=None):
    self.agent.start_episode()
    done = False
    total_reward = 0
    observation = self.env.reset()
    if render:
      self.env.render()
      if not self.renderer_is_initialized:
        self.initialize_renderer()
      if fps:
        secs_per_render = 1 / fps
        last_render_time = time.time()
    for step in itertools.count(1) if max_steps is None else range(1, max_steps + 1):
      action = self.agent.act(observation)
      next_observation, reward, done, _ = self.env.step(action)
      if render:
        if fps:
          time.sleep(max(0, last_render_time + secs_per_render - time.time()))
          last_render_time = time.time()
        self.env.render()
      if train:
        self.agent.receive_reward(reward)
      observation = next_observation
      total_reward += reward
      if print_steps:
        if len(observation.shape) == 1:
          observation_str = ', '.join(f'{x: .5f}' for x in observation)
        else:
          observation_str = '-'
        print(f'Step: {step:>3}   '
              f'Observation: {observation_str}   '
              f'Action: {action}   '
              f'Reward: {reward: >4g}   '
              f'Total reward: {total_reward: >4g}')
      if done:
        break

    self.agent.end_episode(observation, done)
    return total_reward

  def train_agent(self, max_episodes=None, recent_returns_threshold=None,
                  num_recent_returns=200, render_frequency=10,
                  new_line_frequency=10):
    recent_returns = collections.deque(maxlen=num_recent_returns)
    for ep in itertools.count(1) if max_episodes is None else range(1, max_episodes + 1):
      render = render_frequency is not None and ep % render_frequency == 0
      episode_return = self.run_episode(train=True, render=render)
      recent_returns.append(episode_return)
      recent_returns_mean = np.mean(recent_returns)
      print(f'\rEpisode: {ep} '
            f'\tAvg Return: {recent_returns_mean:.3f}',
            end='\n' if ep % new_line_frequency == 0 else '')

      if recent_returns_threshold and recent_returns_mean >= recent_returns_threshold:
        print(f'The recent returns threshold of f{recent_returns_threshold} has been achieved.')
        return

  def demo_agent(self, loop=True, fps=None, delay_between_episodes=2):
    while True:
      episode_return = self.run_episode(render=True, print_steps=True, fps=fps)
      print(f'Episode return: {episode_return}')
      if not loop:
        break
      time.sleep(delay_between_episodes)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='MountainCar-v0')
  parser.add_argument('--alg', type=str, default='vpg')
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--num_procs', type=int, default=None)
  parser.add_argument('--demo', action='store_true', help='demo a trained agent')
  parser.add_argument('--fps', type=int, default=None, help='the fps of the demo')
  # parser.add_argument('--hid', type=int, default=64)
  # parser.add_argument('--l', type=int, default=2)
  # parser.add_argument('--gamma', type=float, default=0.99)
  # parser.add_argument('--steps', type=int, default=4000)
  # parser.add_argument('--epochs', type=int, default=50)
  parser.set_defaults(**ARGS_OVERRIDES)
  args = parser.parse_args()

  # Fork this script if running multiple parallel processes.
  if args.num_procs != 1 and not args.demo:
    mpi.run_parallel_procs(args.num_procs)

  # Set the environment.
  if args.env not in (gym_env.id for gym_env in gym.envs.registry.all()):
    raise ValueError(f'Invalid environment (--env): {args.env}')
  env = gym.make(args.env)

  # Set the agent.
  agent_module = f'algos.{args.alg}'
  if importlib.util.find_spec(agent_module) is None:
    raise ValueError(f'Invalid algorithm (--alg): {args.alg}')
  agent = importlib.import_module(agent_module).Agent(env, args)

  runner = Runner(env=env, agent=agent)

  if args.demo:
    # Demo the agent.
    runner.demo_agent(fps=args.fps)
  else:
    # Train the agent.
    runner.train_agent()


if __name__ == '__main__':
  main()
