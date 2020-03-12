import gym

ENV_IDS = [
    'Breakout-v0',
    'Enduro-v0',
    'MontezumaRevenge-v0',
    'MsPacman-v0',
    'Pong-v0',
    'Qbert-v0',
    'Riverraid-v0',
    'Seaquest-v0',
    'SpaceInvaders-v0',
    'VideoPinball-v0',
    'BipedalWalker-v2',
    'BipedalWalkerHardcore-v2',
    'LunarLander-v2',
    'LunarLanderContinuous-v2',
    'Acrobot-v1',
    'CartPole-v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    'Pendulum-v0',
]


def main():
  for env_id in ENV_IDS:
    env = gym.make(env_id)

    print(env_id)
    if isinstance(env.action_space, gym.spaces.Discrete):
      print(env.observation_space.shape[0], env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
      print(env.observation_space.shape[0], env.action_space.shape[0])
    print()


if __name__ == '__main__':
  main()
