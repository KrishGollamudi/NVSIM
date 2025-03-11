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

# Magnetic field (1 mT along z-axis)
B = np.array([0, 0, 1e-3])  # in Tesla

# Hamiltonian (in units of Hz)
H = D * Sz**2 + E * (Sx**2 - Sy**2) + (g * mu_B / h) * (B[0] * Sx + B[1] * Sy + B[2] * Sz)

# Print Hamiltonian for debugging
print("Hamiltonian:")
print(H)

# Diagonalize the Hamiltonian to find energy levels and eigenstates
eigenvalues, eigenstates = H.eigenstates()

# Print energy eigenvalues (in Hz)
print("Energy eigenvalues (Hz):")
for i, ev in enumerate(eigenvalues):
    print(f"E{i} = {ev:.2e} Hz")

# Print expected eigenvalues
print("Expected eigenvalues (Hz):")
print(f"E0 = 0.00e+00 Hz")
print(f"E1 = {D - E:.2e} Hz")
print(f"E2 = {D + E:.2e} Hz")

# Print Zeeman splitting
zeeman_splitting = g * mu_B * np.linalg.norm(B) / h
print(f"Zeeman splitting: {zeeman_splitting:.2e} Hz")

# Decoherence modeling (T1 and T2)
T1 = 1e-3  # Relaxation time in seconds
T2 = 1e-6  # Decoherence time in seconds
c_ops = [np.sqrt(1/T1) * Sz, np.sqrt(1/T2) * Sx]  # Collapse operators

# Initial state (m_s = 0)
psi0 = basis(3, 1)  # m_s = 0 state

# Time points for simulation
t_list = np.linspace(0, 1e-6, 100)  # 1 microsecond simulation

# Define the expectation operator (projection onto m_s = 0 state)
m0_proj = eigenstates[0] * eigenstates[0].dag()  # Projection operator for m_s = 0

# Solve time evolution with decoherence
result = mesolve(H, psi0, t_list, c_ops, e_ops=[m0_proj])

# Plot 1: Population in m_s = 0 over time
plt.figure(figsize=(10, 6))
plt.plot(t_list, result.expect[0], label="Population in m_s = 0")
plt.xlabel("Time (s)")
plt.ylabel("Population")
plt.title("Time Evolution of m_s = 0 Population")
plt.grid()
plt.legend()

# Simulate ODMR by sweeping microwave frequencies (time-evolution-based approach)
microwave_frequencies = np.linspace(2.8e9, 2.9e9, 1000)  # Frequency sweep around D
PL_intensity_time = np.zeros_like(microwave_frequencies)

for i, freq in enumerate(microwave_frequencies):
    # Apply microwave pulse at frequency freq
    H_mw = (g * mu_B / h) * 1e-3 * Sx  # Microwave Hamiltonian (example amplitude)
    result = mesolve(H + H_mw, psi0, t_list, c_ops, e_ops=[m0_proj])
    PL_intensity_time[i] = result.expect[0][-1]  # Final population in m_s = 0

# Plot 2: ODMR spectrum (time-evolution-based approach)
plt.figure(figsize=(10, 6))
plt.plot(microwave_frequencies, PL_intensity_time, label="ODMR Spectrum (Time Evolution)")
plt.xlabel("Microwave Frequency (Hz)")
plt.ylabel("Photoluminescence Intensity (arb. units)")
plt.title("Simulated ODMR Spectrum of NV Center (Time Evolution)")
plt.grid()
plt.legend()

# Simulate ODMR spectrum (transition-based approach)
PL_intensity_transition = np.zeros_like(microwave_frequencies)

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
        PL_intensity_transition += transition_intensity * np.exp(-((microwave_frequencies - transition_freq)**2) / (2 * (1e6)**2))

# Plot 3: ODMR spectrum (transition-based approach)
plt.figure(figsize=(10, 6))
plt.plot(microwave_frequencies, PL_intensity_transition, label="ODMR Spectrum (Transition-Based)")
plt.xlabel("Microwave Frequency (Hz)")
plt.ylabel("Photoluminescence Intensity (arb. units)")
plt.title("Simulated ODMR Spectrum of NV Center (Transition-Based)")
plt.grid()
plt.legend()

# Simulate Hahn echo
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

# Plot 4: Hahn echo signal
plt.figure(figsize=(10, 6))
plt.plot(tau_values, echo_signal, label="Hahn Echo Signal")
plt.xlabel("Free Evolution Time (s)")
plt.ylabel("Population in m_s = 0")
plt.title("Simulated Hahn Echo Signal")
plt.grid()
plt.legend()

# Magnetic sensitivity analysis
B_values = np.linspace(-1e-3, 1e-3, 100)  # Magnetic field values from -1 mT to +1 mT
frequency_shifts = []

for B in B_values:
    H = D * Sz**2 + E * (Sx**2 - Sy**2) + (g * mu_B / h) * B * Sz
    eigenvalues, _ = H.eigenstates()
    frequency_shifts.append(abs(eigenvalues[2] - eigenvalues[1]))  # Corrected

# Plot 5: Frequency shift vs magnetic field
plt.figure(figsize=(10, 6))
plt.plot(B_values, frequency_shifts, label="Frequency Shift")
plt.xlabel("Magnetic Field (T)")
plt.ylabel("Frequency Shift (Hz)")
plt.title("ODMR Frequency Shift vs Magnetic Field")
plt.grid()
plt.legend()

# Display all plots at once
plt.show()