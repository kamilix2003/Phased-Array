from utils import wavelength

def steer_to_phase(steer_angle, frequency, spacing):
    
    phase_shift = 2 * np.pi * spacing[:, np.newaxis] / wavelength(frequency) * np.sin(steer_angle)
    
    return phase_shift


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    from utils import wavelength
    
    frequency = 2.4e9

    spacing = np.linspace(0.05, 0.5, 10) * wavelength(frequency)
    steer_angle = np.radians(np.linspace(-90, 90, 100))
    
    quant_phase_shift = np.linspace(-180, 180, 16)
    
    phase_shift = steer_to_phase(steer_angle, frequency, spacing)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for i in range(len(spacing)):
        ax.plot(np.degrees(steer_angle), np.degrees(phase_shift[i]), label=f'Spacing: {spacing[i]:.2f} m')


    for q in quant_phase_shift:
        ax.axhline(q, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Steering Angle (degrees)')
    ax.set_ylabel('Phase Shift (degrees)')
    ax.set_title('Phase Shift vs Steering Angle')
    ax.legend()
    plt.show()

