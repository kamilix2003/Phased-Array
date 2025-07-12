import numpy as np
from scipy.constants import c

def linear_to_db(linear: np.ndarray[float]) -> np.ndarray[float]:
    return 10 * np.log10(linear)

def db_to_linear(db: np.ndarray[float]) -> np.ndarray[float]:
    return 10 ** (db / 10)

def config_plot(ax) -> None:
    ax.set_ylim(-50, 0)
    ax.grid(True)
  
import matplotlib.pyplot as plt
    
def plot_array(spacings, weights=None, theta=None, frequency=2.4e9, steer_angles=None):
  from pattern import gen_rect_patter
  from beam_steering import steer_to_phase
  from antenna_array import array_factor, phase_shift
  N = 360
  n = 3
  if weights is None:
    weights = np.zeros_like(spacings)
  if theta is None:
    theta = np.linspace(-np.pi, np.pi, N)
  if steer_angles is None:
    steer_angles = np.linspace(-np.pi/6, np.pi/6, n)
  
  N_elements = spacings.size
  element_pattern = gen_rect_patter(theta, frequency)
  
  d = spacings * c / frequency
  sa = steer_to_phase(N_elements, d, steer_angles, frequency)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(np.ones(N_elements), N_elements, psi)
  ap = np.abs(af * element_pattern)
  ap_db = linear_to_db(ap)
  
  fig = plt.figure()
  ax = plt.subplot(2, 2, (1, 2))
  for i, steer_angle in enumerate(steer_angles):
    ax.plot(theta, ap_db[i, :])
  
  plt.show()
  return fig