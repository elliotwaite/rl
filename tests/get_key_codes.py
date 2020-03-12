import pyglet


def main():
  window = pyglet.window.Window()

  @window.event
  def on_key_press(key_code, modifiers):
    print(f'char: {chr(key_code)}, key_code: {key_code}, modifiers: {modifiers}')

  @window.event
  def on_draw():
    window.clear()

  pyglet.app.run()


if __name__ == '__main__':
  main()
