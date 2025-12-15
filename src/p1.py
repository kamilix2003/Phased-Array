def main():    
  
  import numpy as np
  
  import matplotlib.pyplot as plt
  from scipy.constants import c

  from antenna_array import phase_shift, array_factor
  from pattern import gen_rect_pattern
  
  N = 4
  theta = np.linspace(-np.pi/2, np.pi/2, 360)
  f = np.array([2.4e9, 3e9])
  ep = gen_rect_pattern(theta, f[0])
  
  weights = np.ones(N)
  d = 50e-3 * np.arange(N)
  
  beta = np.zeros((1, N))
  af1 = array_factor(weights, N, 
                    phase_shift(d, f[0], theta, beta)).flatten()
  ap1 = np.abs(af1) * ep
  ap1_db = 20 * np.log10(ap1 / np.max(ap1))
  
  af2 = array_factor(weights, N, 
                    phase_shift(d, f[1], theta, beta)).flatten()
  ap2 = np.abs(af2) * ep
  ap2_db = 20 * np.log10(ap2 / np.max(ap2))
  
  plt.plot(np.degrees(theta), ap1_db)
  plt.plot(np.degrees(theta), ap2_db)
  plt.ylim(-40, 0)
  plt.show()

    
if __name__ == "__main__":
  main()