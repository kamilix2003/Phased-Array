import numpy as np
from scipy.constants import c, epsilon_0, mu_0

class AntennaDesigner:
    
    def __init__(self, epsilon: float, frequency: float, height: float) -> None:
        self._epsilon: float = epsilon
        self.frequency: float = frequency
        self.height: float = height
        pass
    
    def width(self) -> float:
        return c / (2 * self.frequency) * np.sqrt(2 / (self._epsilon + 1))
    
    def epsion_reff(self) -> float:
        return ((self._epsilon + 1) / 2) + \
                ((self._epsilon - 1) / 2) \
                / np.sqrt(1 + 12 * self.height / self.width()) 
                
    def delta_L(self) -> float:
        eps_reff = self.epsion_reff()
        W_h = self.width() / self.height
        numerator = 0.412 * (eps_reff + 0.3) * (W_h + 0.264)
        denominator = (eps_reff - 0.258) * (W_h + 0.8)
        return numerator / denominator
    
    def length(self) -> float:
        eps_reff = self.epsion_reff()
        delta_L = self.delta_L() * self.height
        L = (1 / (2 * self.frequency * np.sqrt(eps_reff * mu_0 * epsilon_0))) - 2 * delta_L
        return L
    
