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
  
def main():
    N_elements = 6
    weights = [1.0, 0.8, 0.9]
    
    generated_weights = gen_weights(N_elements, weights)
    print("Generated Weights:\n", generated_weights)
    
    N_elements = 7
    weights = [1.0, 0.8, 0.9]
    generated_weights = gen_weights(N_elements, weights)
    print("Generated Weights for odd N_elements:\n", generated_weights)
    
if __name__ == "__main__":
    main()
