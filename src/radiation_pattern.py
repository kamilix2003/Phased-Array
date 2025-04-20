import numpy as np

class RadiationPattern:
    
    def __init__(self, 
                 name: str, 
                 frequency: float, 
                 theta: np.ndarray[float],
                 pattern: np.ndarray[complex]) -> None:
        self._name: str = name
        self.frequency: float = frequency
        self.theta: np.ndarray[float] = theta
        self.pattern: np.ndarray[complex] = pattern