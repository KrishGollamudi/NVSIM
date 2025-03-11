import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Constants
D = 2.87e9  # Zero-field splitting in Hz
E = 5e6     # Strain-induced splitting in Hz (example value)
g = 2.0     # Landé g-factor
mu_B = 9.274e-24  # Bohr magneton in J/T
h = 6.626e-34     # Planck's constant in J*s

# Hyperfine coupling constants (for 14N nuclear spin)
A_parallel = 2.16e6  # Parallel hyperfine coupling in Hz
A_perpendicular = 2.16e6  # Perpendicular hyperfine coupling in Hz

# Decoherence times (example values)
T1 = 1e-3  # Relaxation time in seconds
T2 = 1e-6  # Dephasing time in seconds

# Define spin-1 operators for NV center
Sx = jmat(1, 'x')
Sy = jmat(1, 'y')
Sz = jmat(1, 'z')

# Define nuclear spin-1 operators for 14N
Ix = jmat(1, 'x')
Iy = jmat(1, 'y')
Iz = jmat(1, 'z')

# Magnetic field (example: 1 mT along z-axis)
B = np.array([0, 0, 1e-3])  # in Tesla

# Hamiltonian (in units of Hz)
H_ZFS = D * Sz**2 + E * (Sx**2 - Sy**2)  # Zero-field splitting and strain
H_Zeeman = (g * mu_B / h) * (B[0] * Sx + B[1] * Sy + B[2] * Sz)  # Zeeman interaction
H_hyperfine = A_parallel * Sz * Iz + A_perpendicular * (Sx * Ix + Sy * Iy)  # Hyperfine coupling

# Total Hamiltonian
H = H_ZFS + H_Zeeman + H_hyperfine

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

# Simulate ODMR with decoherence
def simulate_odmr(B, microwave_freq, t_list):
    """
    Simulates ODMR by initializing the spin state, applying microwave pulses, and measuring fluorescence.
    """
    # Rebuild Hamiltonian with the input B
    H_local = H_ZFS + (g * mu_B / h) * (B[0] * Sx + B[1] * Sy + B[2] * Sz) + H_hyperfine
    
    # Define collapse operators for decoherence
    c_ops = [
        np.sqrt(1 / T1) * (Sx + 1j * Sy),  # T1 relaxation (ms=0 <-> ms=±1)
        np.sqrt(1 / T2) * Sz,               # T2 dephasing (ms=0 <-> ms=0)
        np.sqrt(1 / T2) * Sx,               # T2 dephasing (ms=±1 <-> ms=±1)
        np.sqrt(1 / T2) * Sy                # T2 dephasing (ms=±1 <-> ms=±1)
    ]
    
    # Define initial state (m_s = 0)
    psi0 = eigenstates[0]  # Ground state (m_s = 0)
    
    # Define microwave driving term (RWA)
    Omega = 1e6  # Rabi frequency in Hz
    H_microwave = [0.5 * Omega * Sx, lambda t, args: np.cos(2 * np.pi * microwave_freq * t) * (0.5 * Omega * Sx)]
    
    # Combine static and time-dependent Hamiltonians
    H_total = [H_local] + H_microwave
    
    # Define expectation operator (population in m_s = 0)
    e_ops = [psi0 * psi0.dag()]  # Must be a list of Qobj
    
    # Solve the master equation
    result = mesolve(H_total, psi0, t_list, c_ops, e_ops)
    
    # Measure population in m_s = 0
    pop_m0 = result.expect[0]
    return pop_m0

# Test ODMR for different microwave frequencies
t_list = np.linspace(0, 1e-6, 100)  # Time points for simulation
odmr_signal = [simulate_odmr(B, freq, t_list)[-1] for freq in microwave_frequencies]

# Plot ODMR signal
plt.figure(figsize=(10, 6))
plt.plot(microwave_frequencies, odmr_signal, label="ODMR Signal")
plt.xlabel("Microwave Frequency (Hz)")
plt.ylabel("Population in m_s = 0")
plt.title("Simulated ODMR Spectrum")
plt.grid()
plt.legend()

# Show both plots
plt.show()

# Magnetic Sensitivity Analysis
def estimate_sensitivity(delta_nu, T2_star):
    """
    Estimates magnetic sensitivity using the formula:
    eta ~ delta_nu / (gamma * sqrt(T2_star))
    """
    gamma = g * mu_B / h  # Gyromagnetic ratio in Hz/T
    sensitivity = delta_nu / (gamma * np.sqrt(T2_star))
    return sensitivity

# Example: Compute sensitivity for a 1 MHz frequency shift and T2* = 1 µs
delta_nu = 1e6  # Frequency shift in Hz
T2_star = 1e-6  # Coherence time in seconds
sensitivity = estimate_sensitivity(delta_nu, T2_star)
print(f"Magnetic Sensitivity: {sensitivity:.2e} T/√Hz")
