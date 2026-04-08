import numpy as np
import matplotlib.pyplot as plt

def main():
  
  f = np.arange(2e9, 4e9, .1e9)
  c = 3e8
  
  spacing = 50e-3
  
  rel_freq = f * spacing / c
  
  fig = plt.figure(figsize=(6, 5))
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(f / 1e9, rel_freq, linestyle='-', color='C0')
  ax.set_xlim(2, 4)
  ax.set_xticks(np.arange(2, 4, 0.1))
  ax.set_ylim(0, 1)
  ax.set_yticks(np.arange(0, 1.1, 0.1))
  ax.grid()
  ax.set_xlabel("Frequency (GHz)")
  ax.set_ylabel("Relative Spacing (λ)")
  ax.set_title(f"Relative Spacing vs Frequency (d={spacing*1e3} mm)")
  plt.show()  
  
  for i in f:
    print(f"Frequency: {i/1e9:.2f} GHz, Relative Spacing: {i*spacing/c:.3f} λ")
  
if __name__ == "__main__":
  main()