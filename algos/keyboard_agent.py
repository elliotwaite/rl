import argparse
import collections
import time

import gym
import numpy as np

from utils import env_window

UP_ARROW_KEY_CODES = 273, 65362
DOWN_ARROW_KEY_CODES = 274, 65364
LEFT_ARROW_KEY_CODES = 276, 65361
RIGHT_ARROW_KEY_CODES = 275, 65363

DEFAULT_KEY_MAP = {
    (): 0,
    (ord('1'),): 1,
    (ord('2'),): 2,
    (ord('3'),): 3,
    (ord('4'),): 4,
    (ord('5'),): 5,
    (ord('6'),): 6,
    (ord('7'),): 7,
    (ord('8'),): 8,
    (ord('9'),): 9,
}
LEFT_RIGHT_KEY_MAP = {
    (): 0,
    (ord('d'),): 1,
}
LEFT_NONE_RIGHT_KEY_MAP = {
    (ord('a'),): 0,
    (): 1,
    (ord('d'),): 2,
}
NONE_LEFT_UP_RIGHT_KEY_MAP = {
    (): 0,
    (ord('a'),): 1,
    (ord('w'),): 2,
    (ord('d'),): 3,
}


def build_continuous_key_map_recursively(
    key_map, num_actions, key_rows, action_values, keys, action_indexes, action=0):
  """Used by `continuous_key_map` to recursively build a key map.

  Adds the keys for the current action row of keys to the key map dictionary.
  """
  for i, char in enumerate(key_rows[action]):
    new_keys = keys + ([ord(char)] if i else [])
    new_action_indexes = action_indexes + [(action, i)]
    key_map[tuple(new_keys)] = [action_values[action_index]
                                for action_index in new_action_indexes]
    if action < num_actions - 1:
      build_continuous_key_map_recursively(
          key_map, num_actions, key_rows, action_values, keys=new_keys,
          action_indexes=new_action_indexes, action=action + 1)


def continuous_key_map(action_space):
  """Returns a continuous key map.

  Maps the the keys on the keyboard to action values. The keys from left to
  right map to a linear range of action values from min to max. Only 10 keys
  from each row are used. The top row (the numbers row, from the 1 key to 0
  key) are for the first action, the second row for the second action, the
  third row for the third action, and the fourth row for the fourth action.
  If the environment has more than 4 action values, an error is raised.
  When a key is not pressed for an action, that action's value will be 0.
  """
  if len(action_space.shape) == 1 and action_space.shape[0] <= 4:
    num_actions = action_space.shape[0]
    print(f'Number of continuous actions: {num_actions}')
    print(f'Min action values: {action_space.low}')
    print(f'Max action values: {action_space.high}')
    key_map = {}
    key_rows = [' 1234567890', ' qwertyuiop', ' asdfghjkl;', ' zxcvbnm,./']
    action_values = np.concatenate((
        np.zeros((1, num_actions)),
        np.linspace(action_space.low, action_space.high, num=10))).T
    keys = []
    action_indexes = []
    build_continuous_key_map_recursively(
        key_map, num_actions, key_rows, action_values, keys, action_indexes)
    return key_map
  else:
    raise ValueError('Unsupported environment action space.')


def standardized_key_code(key_code, env):
  """Translates non-standard directional key codes into standardized
  directional key codes, leaving all other key codes unchanged.

  The standard keys for directional actions are the 'w', 'a', 's', and 'd'
  keys. This function allows you to also use the 'i', 'j', 'k', and 'l' keys,
  as well as the arrow keys, for directional actions by translating the key
  codes for those keys in the key codes of the standard directional keys.
  """
  if not isinstance(env.action_space, gym.spaces.Discrete):
    return key_code
  if key_code in (ord('i'), *UP_ARROW_KEY_CODES):
    return ord('w')
  if key_code in (ord('j'), *LEFT_ARROW_KEY_CODES):
    return ord('a')
  if key_code in (ord('k'), *DOWN_ARROW_KEY_CODES):
    return ord('s')
  if key_code in (ord('l'), *RIGHT_ARROW_KEY_CODES):
    return ord('d')
  return key_code


def env_to_key_map(env):
  """Returns a key map for the environment.

  A key map is an ordered dictionary where the dictionary keys are tuples of
  the key codes of the keys that need to be pressed to trigger the action, and
  the dictionary values are the action values. The dictionary is ordered by
  descending length of the key tuples, so that actions that require more keys
  to be pressed simultaneously are listed before actions that require fewer
  keys to be pressed simultaneously.
  """
  if isinstance(env.action_space, gym.spaces.Discrete):
    try:
      key_map = env.get_keys_to_action()
    except AttributeError:
      if env.action_space.n == 2:
        key_map = LEFT_RIGHT_KEY_MAP
      elif env.action_space.n == 3:
        key_map = LEFT_NONE_RIGHT_KEY_MAP
      elif env.action_space.n == 4:
        key_map = NONE_LEFT_UP_RIGHT_KEY_MAP
      else:
        print('No custom key map found. Using the default key map '
              '(number keys map to their respective actions).')
        key_map = DEFAULT_KEY_MAP
  elif isinstance(env.action_space, gym.spaces.Box):
    key_map = continuous_key_map(env.action_space)
  else:
    raise ValueError('Unsupported environment action space.')

  # Order and standardize the key map.
  keys = list(key_map.keys())
  keys.sort(key=lambda key: len(key), reverse=True)
  return collections.OrderedDict(
      (tuple(standardized_key_code(key_code, env) for key_code in key), key_map[key])
      for key in keys)


class KeyboardAgent:
  def __init__(self, env):
    self.env = env

    self.key_map = env_to_key_map(env)
    self.pressed_keys = set()
    self.action = self.key_map[()]

    self.env.reset()
    self.env.render()
    self.env.unwrapped.viewer.window.on_key_press = self.on_key_press
    self.env.unwrapped.viewer.window.on_key_release = self.on_key_release

  def on_key_press(self, key_code, modifiers):
    key_code = standardized_key_code(key_code, self.env)
    self.pressed_keys.add(key_code)
    self.update_action()

  def on_key_release(self, key_code, modifiers):
    key_code = standardized_key_code(key_code, self.env)
    self.pressed_keys.discard(key_code)
    self.update_action()

  def update_action(self):
    for keys, action in self.key_map.items():
      if set(keys).issubset(self.pressed_keys):
        self.action = action
        return

  def act(self, obs):
    return self.action


def demo_agent(env, agent, fps=60, delay_between_episodes=2):
  secs_per_render = 1 / fps
  env_window.setup_env_window(env)
  while True:
    step = 1
    total_rew = 0
    done = False
    obs = env.reset()
    last_render_time = time.time()
    env.render()
    while not done:
      act = agent.act(obs)
      obs, rew, done, _ = env.step(act)
      total_rew += rew
      sleep_time = secs_per_render - (time.time() - last_render_time)
      if sleep_time > 0:
        time.sleep(sleep_time)
      last_render_time = time.time()
      env.render()
      if len(obs.shape) == 1:
        obs_str = ', '.join(f'{x: .5f}' for x in obs)
      else:
        obs_str = '-'
      print(f'Step: {step:>3}   '
            f'Observation: {obs_str}   '
            f'Action: {act}   '
            f'Reward: {rew: >4g}   '
            f'Total reward: {total_rew: >4g}')
      step += 1

    print(f'Episode return: {total_rew}')
    time.sleep(delay_between_episodes)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='MountainCar-v0')
  parser.add_argument('--fps', type=float, default=30, help='Frames per second')
  parser.add_argument('--delay_between_episodes', type=float, default=2, help='Seconds of delay between episodes')

  parser.set_defaults(**dict(
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
      # env='Acrobot-v1',
      # env='CartPole-v1',
      # env='MountainCar-v0',
      # env='MountainCarContinuous-v0',
      # env='Pendulum-v0',

      # Other
      # env='FlappyBird-v0',
  ))

  args = parser.parse_args()

  env = gym.make(args.env)
  agent = KeyboardAgent(env)

  demo_agent(env, agent, args.fps, args.delay_between_episodes)


if __name__ == '__main__':
  main()
