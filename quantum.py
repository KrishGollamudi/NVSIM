import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Constants
D = 2.87e9  # Zero-field splitting in Hz
E = 5e6     # Strain-induced splitting in Hz (example value)
g = 2.0     # Landé g-factor
mu_B = 9.274e-24  # Bohr magneton in J/T
h = 6.626e-34     # Planck's constant in J*s

# Define spin-1 operators
Sx = jmat(1, 'x')
Sy = jmat(1, 'y')
Sz = jmat(1, 'z')

# Magnetic field (example: 1 mT along z-axis)
B = np.array([0, 0, 1e-3])  # in Tesla

# Hamiltonian (in units of Hz)
H = D * Sz**2 + E * (Sx**2 - Sy**2) + (g * mu_B / h) * (B[0] * Sx + B[1] * Sy + B[2] * Sz)

# Diagonalize the Hamiltonian to find energy levels and eigenstates
eigenvalues, eigenstates = H.eigenstates()

# Print energy eigenvalues (in Hz)
print("Energy eigenvalues (Hz):")
for i, ev in enumerate(eigenvalues):
    print(f"E{i} = {ev:.2e} Hz")

# Simulate ESR spectrum
microwave_frequencies = np.linspace(2.8e9, 2.9e9, 1000)  # Frequency sweep around D
PL_intensity = np.zeros_like(microwave_frequencies)

# Calculate transition frequencies and intensities
for i in range(len(eigenvalues)):
    for j in range(i + 1, len(eigenvalues)):
        transition_freq = abs(eigenvalues[j] - eigenvalues[i])  # Transition frequency in Hz
        # Calculate transition dipole moment
        transition_matrix_element = (eigenstates[j].dag() * Sx * eigenstates[i])
        # Ensure transition_matrix_element is a Qobj
        if isinstance(transition_matrix_element, Qobj):
            transition_intensity = abs(transition_matrix_element.full()[0][0])  # Extract the scalar value
        else:
            transition_intensity = abs(transition_matrix_element)  # Fallback for scalar values
        # Add Gaussian broadening to the transition
        PL_intensity += transition_intensity * np.exp(-((microwave_frequencies - transition_freq)**2) / (2 * (1e6)**2))

# Plot ESR spectrum
plt.figure(figsize=(10, 6))
plt.plot(microwave_frequencies, PL_intensity, label="ESR Spectrum")
plt.xlabel("Microwave Frequency (Hz)")
plt.ylabel("Photoluminescence Intensity (arb. units)")
plt.title("Simulated ESR Spectrum of NV Center")
plt.grid()
plt.legend()
plt.show()

# Simulate dynamical decoupling (Hahn echo)
def simulate_hahn_echo(B, tau):
    """
    Simulates the Hahn echo sequence for a given magnetic field B and free evolution time tau.
    """
    # Define the initial state (m_s = 0)
    psi0 = eigenstates[0]  # Ground state (m_s = 0)

    # Define the pi/2 pulse (rotates spin to superposition state)
    U_pi2 = (-1j * np.pi/2 * Sx).expm()

    # Define the pi pulse (flips spin states)
    U_pi = (-1j * np.pi * Sx).expm()

    # Free evolution under the Hamiltonian
    U_free = (-1j * H * tau).expm()

    # Apply the Hahn echo sequence: pi/2 - tau/2 - pi - tau/2 - pi/2
    psi = U_pi2 * psi0  # Initial pi/2 pulse
    psi = U_free * psi  # Free evolution for tau/2
    psi = U_pi * psi    # Pi pulse
    psi = U_free * psi  # Free evolution for tau/2
    psi = U_pi2 * psi   # Final pi/2 pulse

    # Measure the population in m_s = 0
    pop_m0 = abs((eigenstates[0].dag() * psi).full()[0][0])**2
    return pop_m0

# Test Hahn echo for different free evolution times
tau_values = np.linspace(0, 2e-6, 100)  # Free evolution times from 0 to 2 microseconds
echo_signal = [simulate_hahn_echo(B, tau) for tau in tau_values]

# Plot Hahn echo signal
plt.figure(figsize=(10, 6))
plt.plot(tau_values, echo_signal, label="Hahn Echo Signal")
plt.xlabel("Free Evolution Time (s)")
plt.ylabel("Population in m_s = 0")
plt.title("Simulated Hahn Echo Signal")
plt.grid()
plt.legend()
plt.show()
