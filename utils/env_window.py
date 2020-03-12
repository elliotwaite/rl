def setup_env_window(env, make_closable=True, fit_to_screen=True):
  env.reset()
  env.render()

  if make_closable:
    # Make the close button on the renderer's window exit the program.
    env.env.viewer.window.on_close = exit

  if fit_to_screen:
    # If the renderer's window is resizable, resize it to be as large as
    # possible while preserving it's aspect ratio and centering its position.
    if env.env.viewer.window.resizeable:
      screen_width = env.env.viewer.window.screen.width
      screen_height = env.env.viewer.window.screen.height
      aspect_ratio = env.env.viewer.window.width / env.env.viewer.window.height
      new_width = int(screen_height * aspect_ratio)
      new_height = screen_height
      if new_width > screen_width:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
      env.env.viewer.window.set_location(int((screen_width - new_width) / 2),
                                         int((screen_height - new_height) / 2))
      env.env.viewer.window.width = new_width
      env.env.viewer.window.height = new_height
