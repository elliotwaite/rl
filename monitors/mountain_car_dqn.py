import matplotlib.pyplot as plt


class MountainCarDQNMonitor:
  def __init__(self, agent):
    self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
    self.fig.canvas.set_window_title('Mountain Car Monitor')

    xlim, ylim = zip(
        agent.env.observation_space.low,
        agent.env.observation_space.high)

    # 2D state.
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)
    self.ax.set_xlabel('Position')
    self.ax.set_ylabel('Velocity')
    self.ax.plot([0.5, 0.5], self.ax.get_ylim(), color='#cccc00')
    self.state_point_2d, = self.ax.plot(-0.5, 0, marker='o', color='black')

    # Make pressing the close button on the plot exit the python script.
    self.fig.canvas.mpl_connect('close_event', exit)

    plt.ion()
    plt.show()

  def update(self, agent, state, action, reward, next_state, done):
    self.state_point_2d.set_data(state[0], state[1])

    # Push updates to the canvas.
    self.fig.canvas.draw()
    self.fig.canvas.start_event_loop(0.001)
