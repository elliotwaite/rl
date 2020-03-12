class BaseAgent:
  """The base class for all agents."""

  def set_is_training(self, is_training):
    """Sets whether the agent should be in training mode or eval/demo mode.

    Args:
      is_training (bool): If True, the agent should be set to training mode.
          If False, the agent should be set to eval/demo mode.
    """
    pass

  def start_episode(self):
    """Called at the beginning of each episode."""
    pass

  def act(self, obs):
    """Chooses an action for the given observation.

    This is the only required method, all others are optional.

    Args:
      obs (Numpy Array): An observation value.

    Returns: An action value (the shape and type depends on the environment).
    """
    raise NotImplemented

  def receive_reward(self, rew):
    """Receives the most recent reward.

    Args:
      rew (float): The reward value received after the most recent action.
    """
    pass

  def end_episode(self, obs, done):
    """Called at the end of each episode.

    Args:
      obs (Numpy array): The last observation after the most recent action.
      done (bool): The last done value of the episode. If False, the episode
          was ended because it exceeded the maximum specified steps.
    """
    pass
