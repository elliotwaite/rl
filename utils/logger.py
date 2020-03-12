def log_step(step, obs, act, rew, total_rew):
  if len(obs.shape) == 1:
    obs_str = ', '.join(f'{x: .5f}' for x in obs)
  else:
    obs_str = '-'
  print(f'Step: {step:>3}   '
        f'Observation: {obs_str}   '
        f'Action: {act}   '
        f'Reward: {rew: >4g}   '
        f'Total reward: {total_rew: >4g}')