
import numpy as np

def gen_weights(N_elements, weights):
    
    if len(weights) == 0:
        return np.ones(N_elements)
    
    if len(weights) != N_elements // 2:
        raise ValueError("Length of weights must be equal to N_elements // 2")
    
    out = np.zeros(N_elements)
    center_idx = N_elements // 2
    if N_elements % 2 == 0:
      for i in range(N_elements // 2):
          out[center_idx + i] = weights[i]
          out[center_idx - i - 1] = weights[i]
    else:
      out[center_idx] = weights[0]
      for i in range(1, N_elements // 2 + 1):
          out[center_idx + i] = weights[i - 1]
          out[center_idx - i] = weights[i - 1]
    
    return out
  
def generate_all_windows(n_elements, window_kwargs=None):
    import inspect
    from scipy.signal import windows as sp_windows
    if window_kwargs is None:
        window_kwargs = {}
    result = {}
    # Get all callables in scipy.signal.windows that take at least one positional arg
    for name, func in inspect.getmembers(sp_windows, inspect.isfunction):
        # Only include functions that have n as first arg
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params or params[0].name not in ('M', 'N', 'numtaps', 'n', 'window_len'):
            continue
        # Prepare kwargs for this window
        kwargs = window_kwargs.get(name, {})
        try:
            arr = func(n_elements, **kwargs)
            result[name] = arr
        except Exception:
            # Skip windows that error out for given n_elements/kwargs
            continue
    return result
    
  
def main():
    N_elements = 6
    weights = [1.0, 0.8, 0.9]
    
    generated_weights = gen_weights(N_elements, weights)
    print("Generated Weights:\n", generated_weights)
    
    N_elements = 7
    weights = [1.0, 0.8, 0.9]
    generated_weights = gen_weights(N_elements, weights)
    print("Generated Weights for odd N_elements:\n", generated_weights)
    
    from matplotlib import pyplot as plt
    from scipy.signal import windows
    
    tukey =  windows.tukey(N_elements, alpha=0.5)
    HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]
    weights = windows.general_cosine(N_elements, HFT90D)
    
    plt.plot(weights, label='tukey')
    plt.legend()
    plt.grid()
    plt.show()
    
    
if __name__ == "__main__":
    main()
