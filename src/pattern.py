import numpy as np
from scipy.constants import c

def gen_rect_pattern(theta, freq, er=3.35):
  W = c / 2 / freq * np.sqrt(2 / (er + 1))
  return (np.cos(theta) * np.sinc(2 * np.pi * c / freq * W / 2 * np.sin(theta)) + 1) / 2

def main():
  
  import matplotlib.pyplot as plt
  
  theta = np.linspace(-np.pi, np.pi, 360)
  pattern = gen_rect_pattern(theta, 2.4e9)
  fig = plt.figure()
  # plt.plot(theta, pattern)
  plt.polar(theta, pattern)
  plt.show()
  pass
  
if __name__ == "__main__":
  main()