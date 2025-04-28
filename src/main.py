import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

from antenna_element import AntennaElement
from radiation_pattern import RadiationPattern
from antenna_array import AntennaArray
from utils import wavelength, linear_to_db, config_plot

N = 1000
theta = np.linspace(-np.pi, np.pi, N)
frequency = 2.4e9  # Frequency in Hz

pattern1 = RadiationPattern(
    name='Pattern1',
    frequency=frequency,
    theta=theta,
    # pattern=np.cos(theta/2 + np.pi/4)**2  # Example pattern
    pattern=np.sinc(theta - np.pi/2)**2  # Example pattern
)

element1 = AntennaElement(
    name='Element1',
    patterns=np.array([pattern1])
)

array = AntennaArray(
    name='Array1',
    antenna=element1,
    num_elements=4,
    spacings=wavelength(frequency) * 0.25  # Spacing in meters
)

beta = 0

af = array.array_factor(frequency, theta, beta)

E = array.radiation_pattern(frequency, theta, beta)

fig = plt.figure(figsize=(10, 6))
polar = True
ax = plt.subplot(131, polar = polar)
config_plot(ax, polar)
plt.plot(theta, linear_to_db(np.abs(af)), label='Array Factor')
plt.title('Array Factor')

ax = plt.subplot(132, polar = polar)
config_plot(ax, polar)
plt.plot(pattern1.theta, linear_to_db(pattern1.pattern), label='Element Pattern')
plt.title('Element Pattern')

ax = plt.subplot(133, polar = polar)
config_plot(ax, polar)
plt.plot(theta, linear_to_db(np.abs(E)), label='Radiation Pattern')
plt.plot(theta, np.abs(E), label='Radiation Pattern')
plt.title('Antenna Array Pattern')

from pattern_measurements import find_maximas, find_nulls

i = find_nulls(np.abs(E))
j = find_maximas(np.abs(E))
print(i, j)
plt.plot(theta[j], linear_to_db(np.abs(E[j])), 'go', label='Maximas')

for k in i:
    plt.plot([theta[k], theta[k]], ax.get_ylim(), 'r--', label='Nulls')


plt.show()