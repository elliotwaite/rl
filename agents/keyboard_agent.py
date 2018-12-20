import agents


class KeyboardAgent(agents.BaseAgent):
  def __init__(self,
               default_action=0,
               key_to_action_map=None,
               **kwargs):
    super().__init__(**kwargs)
    self.action = self.default_action = default_action
    self.key_to_action_map = key_to_action_map

    self.env.unwrapped.viewer.window.on_key_press = self.key_press
    self.env.unwrapped.viewer.window.on_key_release = self.key_release

  def act(self, state):
    return self.action

  def key_press(self, key_code, mod):
    key = int(key_code - ord('0'))
    if self.key_to_action_map is not None:
      if key in self.key_to_action_map:
        self.action = self.key_to_action_map[key]
    elif 0 <= key <= self.num_actions:
      self.action = key

  def key_release(self, key_code, mod):
    key = int(key_code - ord('0'))
    if self.key_to_action_map is not None:
      if (key in self.key_to_action_map and
          self.action == self.key_to_action_map[key]):
        self.action = self.default_action
    elif 0 <= key <= self.num_actions and self.action == key:
      self.action = self.default_action
