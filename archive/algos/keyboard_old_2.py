import time

import gym
import numpy as np

from algos.base_agent import BaseAgent

"""
symbol: 32, space
symbol: 119, W
symbol: 97, A
symbol: 115, S
symbol: 100, D
"""

LEFT = 65361
UP = 65362
RIGHT = 65363
DOWN = 65364


def translate_directional_key_code(code):
  if code in (ord('i'), UP):
    return ord('w')
  if code in (ord('j'), LEFT):
    return ord('a')
  if code in (ord('k'), DOWN):
    return ord('s')
  if code in (ord('l'), RIGHT):
    return ord('d')
  return code

#
# print(ord('a'))
# print(ord('A'))
# exit()
# SYMBOL_TO_KEY = {
#     32: 'SPACE',
#     48: '0',
#     49: '1',
#     50: '2',
#     51: '3',
#     52: '4',
#     53: '5',
#     54: '6',
#     55: '7',
#     56: '8',
#     57: '9',
#     97: 'A',
#     100: 'D',
#     105: 'I',
#     106: 'J',
#     107: 'K',
#     108: 'L',
#     115: 'S',
#     119: 'W',
#     65361: 'LEFT',
#     65362: 'UP',
#     65363: 'RIGHT',
#     65364: 'DOWN',
# }

# Key maps.
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
LEFT_RIGHT = {
    (): 0,
    (ord('d')): 1,
}
LEFT_NONE_RIGHT = {
    (ord('a')): 0,
    (): 1,
    (ord('d')): 2,
}


#
# NONE_FIRE_RIGHT_LEFT = dict(
#     default_action=0,
#     key_actions={
#         'SPACE': 1,
#         'RIGHT': 2,
#         'D': 2,
#         'L': 2,
#         'LEFT': 3,
#         'A': 3,
#         'J': 3,
#     })
# NONE_FIRE_UP_LEFT_RIGHT_DOWN = dict(
#     default_action=0,
#     key_actions={
#         'SPACE': 1,
#         'UP': 2,
#         'W': 2,
#         'I': 2,
#         'LEFT': 3,
#         'A': 3,
#         'J': 3,
#         'RIGHT': 4,
#         'D': 4,
#         'L': 4,
#         'DOWN': 5,
#         'S': 5,
#         'K': 5,
#     })


def get_continuous_key_map(action_space):
  if action_space.shape == (1,):
    actions = np.linspace(action_space.low, action_space.high, num=10)
    return {
        (): [0],
        (ord('1')): actions[0],
        (ord('2')): actions[1],
        (ord('3')): actions[2],
        (ord('4')): actions[3],
        (ord('5')): actions[4],
        (ord('6')): actions[5],
        (ord('7')): actions[6],
        (ord('8')): actions[7],
        (ord('9')): actions[8],
        (ord('0')): actions[9],
    }
  else:
    raise ValueError('Unknown action space.')


def env_to_key_map(env):
  # if env.spec.id in ENV_ID_TO_KEY_MAP:
  #   return ENV_ID_TO_KEY_MAP
  if isinstance(env.action_space, gym.spaces.Discrete):
    a = env.get_keys_to_action()
    if env.action_space.n == 2:
      return LEFT_RIGHT
    if env.action_space.n == 3:
      return LEFT_NONE_RIGHT
    if env.action_space.n == 4:
      return NONE_FIRE_RIGHT_LEFT
    if env.action_space.n == 6:
      return NONE_FIRE_UP_LEFT_RIGHT_DOWN
  elif isinstance(env.action_space, gym.spaces.Box):
    return get_continuous_key_map(env.action_space)
  else:
    raise ValueError('Unknown key map.')

  return DEFAULT_KEY_MAP


class Agent(BaseAgent):
  def __init__(self, env, args):
    self.env = env

    key_map = env_to_key_map(env)
    self.default_action = key_map['default_action']
    self.key_actions = key_map['key_actions']

    self.action = self.default_action
    self.env.reset()
    self.env.render()
    self.env.unwrapped.viewer.window.on_key_press = self.on_key_press
    self.env.unwrapped.viewer.window.on_key_release = self.on_key_release

  def act(self, obs):
    return self.action

  def on_key_press(self, symbol, modifiers):
    if symbol not in SYMBOL_TO_KEY:
      return
    key = SYMBOL_TO_KEY[symbol]
    if key in self.key_actions:
      self.action = self.key_actions[key]

  def on_key_release(self, symbol, modifiers):
    if symbol not in SYMBOL_TO_KEY:
      return
    key = SYMBOL_TO_KEY[symbol]
    if key in self.key_actions and self.action == self.key_actions[key]:
      self.action = self.default_action

  def run_episode(self):
    total_rew = 0
    obs = self.env.reset()
    done = False
    self.env.render()
    step = 0
    while not done:
      action = self.act()
      next_obs, rew, done, _ = self.env.step(action)
      total_rew += rew
      self.env.render()
      obs_str = ', '.join(f'{x: .5f}' for x in obs)
      print(f'Step: {step + 1:>3}   '
            f'Action: {action}   '
            f'State: {obs_str}   '
            f'Reward: {rew: >4g}   '
            f'Total Reward: {total_rew: >4g}')
      obs = next_obs
      step += 1

  def demo(self):
    while True:
      self.run_episode()
      time.sleep(2)
