import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

def gen_rect_patter(theta, freq, er=3.35):
  W = c / 2 / freq * np.sqrt(2 / (er + 1))
  return np.cos(theta) * np.sinc(2 * np.pi * c / freq * W / 2 * np.sin(theta))

def main():
  theta = np.linspace(-np.pi, np.pi, 360)
  pattern = gen_rect_patter(theta, 2.4e9)
  fig = plt.figure()
  # plt.plot(theta, pattern)
  plt.polar(theta, pattern)
  plt.show()
  pass
  
if __name__ == "__main__":
  main()