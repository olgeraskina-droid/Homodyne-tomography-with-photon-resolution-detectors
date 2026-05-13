import numpy as np

# Measurement settings
count_theta = 30
iterations = 1000
d = 10          # Full Hilbert space dimension (signal + LO)
dim_s = 3       # Signal subspace dimension

# Known (assumed) calibration parameters
loss_KNOW = [0.55] * 8
T_know = [0.5] * 7
alpha_LO_theory = 2.0
theta_array = np.linspace(0, 2 * np.pi, count_theta)

# Squeezing parameter
R = 1