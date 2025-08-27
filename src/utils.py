import numpy as np
from scipy.constants import c

def linear_to_db(linear: np.ndarray[float]) -> np.ndarray[float]:
    return 10 * np.log10(linear)

def db_to_linear(db: np.ndarray[float]) -> np.ndarray[float]:
    return 10 ** (db / 10)
  
def get_pattern(theta, frequency, num_elements, d, beta, weights=None):
  from pattern import gen_rect_pattern
  from antenna_array import array_factor, phase_shift
  
  if weights is None:
    weights = np.ones(num_elements)
  psi = phase_shift(d, frequency, theta, beta)
  af = array_factor(weights, num_elements, psi)
  ap = np.abs(af)
  return gen_rect_pattern(theta, frequency) * ap