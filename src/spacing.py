import numpy as np

def gen_spacing(N_elements, spacings):
  if len(spacings) == 1:
    return uniform_spacing(N_elements, spacings[0])
  if N_elements % 2 == 0:
    return np.concatenate((-np.cumsum(spacings)[::-1], np.cumsum(spacings))) + np.sum(spacings)
  else:
    return np.concatenate((-np.cumsum(spacings)[::-1], [0], np.cumsum(spacings))) + np.sum(spacings)

def uniform_spacing(N_elements, spacing):
  return np.cumsum(np.full(N_elements, spacing)) - spacing

def test():
  test_spacings = [0.1, .2, .3]
  N = 7
  print(gen_spacing(N, test_spacings))
  print(gen_spacing(N-1, test_spacings))
  print(uniform_spacing(N, 0.1))
  pass

if __name__ == "__main__":
  test()