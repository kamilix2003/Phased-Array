import numpy as np

def gen_spacing(N_elements, spacings):
  temp_element = 0
  if N_elements % 2 == 0:
    temp_element = 1
  out = np.zeros(N_elements + temp_element)
  center_idx = (N_elements + temp_element) // 2
  for i in range(1, center_idx+1):
    out[center_idx + i] = np.sum(spacings[0:i])
    out[center_idx - i] = - np.sum(spacings[0:i])
  if temp_element == 1:
    return out[1:]
  return out
  pass

def test():
  test_spacings = [0.1, 0.2, 0.5]
  N = 5
  print(gen_spacing(N, test_spacings))
  pass

if __name__ == "__main__":
  test()