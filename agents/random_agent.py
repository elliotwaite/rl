import agents


class RandomAgent(agents.BaseAgent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def act(self, state):
    action = self.env.action_space.sample()
    return action
