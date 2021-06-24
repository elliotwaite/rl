import algos
import monitors


def demo_keyboard_agent():
  agent = algos.KeyboardAgent(
      env_id='MountainCar-v0',
      default_action=1,
      key_to_action_map={
          1: 0,
          2: 2
      })
  agent.demo_agent()


def demo_random_agent():
  agent = algos.RandomAgent(
      env_id='MountainCar-v0',
      monitor_class=monitors.MountainCarDQNMonitor)
  agent.demo_agent()


def train_dqn_1_step_agent():
  agent = algos.DQN1StepAgent(
      env_id='MountainCar-v0')
  agent.train_agent()


def main():
  demo_keyboard_agent()
  # demo_random_agent()
  # train_dqn_1_step_agent()


if __name__ == '__main__':
  main()
