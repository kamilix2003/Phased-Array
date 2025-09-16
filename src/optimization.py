from scipy.constants import c
import numpy as np

from antenna_array import array_factor, phase_shift
from beam_steering import gen_steer_directions
from spacing import gen_spacing

from cost import total_cost

def get_log_dict():
  return {
    "iteration": [],
    "bandwidth": [],
    "spacing": [],
    "weights": [],
    "d": [],
    "total_cost": [],
    "cost": {
      "sll": [],
      "cov": [],
    }
  }

def optimize_pattern(x,
                     theta,
                     element_pattern,
                     N_elements,
                     scan_step,
                     frequency,
                     log:dict=None):

  
  spacing = np.array([x[0]])
  
  d = gen_spacing(N_elements, spacing) * c / frequency
  weights = np.ones(N_elements)
  weights = np.array(x[1:]) if len(x) > 1 else weights
  
  sa = gen_steer_directions(N_elements, step=scan_step, extend=1)
  psi = phase_shift(d, frequency, theta, sa);
  af = array_factor(weights, N_elements, psi)
  ap = np.abs(af) * element_pattern
  
  out = total_cost(theta, ap, log)
  
  if log is not None:
    log["iteration"].append(len(log["iteration"])+1)
    log["spacing"].append(spacing[0])
    log["weights"].append(weights)
    log["d"].append(d)
  
  return out