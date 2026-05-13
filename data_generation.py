import numpy as np
import random
from config import count_theta, iterations, d, dim_s
from utils import density_matrix_coherent_lo
from povm import calculate_q_with_loss, ubs

def simulate_tree_fast(n_arr, q_eff):
    """Fast simulation of tree detector clicks given n photons."""
    probs = np.concatenate([q_eff, [1.0 - np.sum(q_eff)]])  # length 9, sum = 1
    k = np.zeros(len(n_arr), dtype=int)
    for idx, n in enumerate(n_arr):
        if n == 0:
            k[idx] = 0
            continue
        choices = np.random.choice(9, size=n, p=probs)
        fired = choices[choices < 8]
        if len(fired) > 0:
            k[idx] = len(np.unique(fired))
        else:
            k[idx] = 0
    return k

def create_data(ro_s_0, alpha_lo_real, R, t1, t2, loss1_real, loss2_real):
    """Generate synthetic measurement data for PVM-like tree detector."""
    ro_s = np.zeros((d, d), dtype=complex)
    ro_s[:dim_s, :dim_s] = ro_s_0

    q1_eff = calculate_q_with_loss(t1, loss1_real)
    q2_eff = calculate_q_with_loss(t2, loss2_real)
    u = ubs(d, R)
    counts = np.zeros((count_theta, 9, 9), dtype=int)
    theta_array_real = np.linspace(0, 2 * np.pi, count_theta)

    for j, theta in enumerate(theta_array_real):
        theta = random.uniform(theta - np.pi / 17, theta + np.pi / 17)
        ro_lo_real = density_matrix_coherent_lo(d, alpha_lo_real, theta)
        ro_in_real = np.kron(ro_lo_real, ro_s)
        ro_out_real = u @ ro_in_real @ u.conj().T
        probs_real = np.maximum(np.real(np.diag(ro_out_real)), 0)
        probs_real /= probs_real.sum()
        indices = np.random.choice(len(probs_real), size=iterations, p=probs_real)
        n1_vals = indices // d
        n2_vals = indices % d

        k1_vals = simulate_tree_fast(n1_vals, q1_eff)
        k2_vals = simulate_tree_fast(n2_vals, q2_eff)
        for k1, k2 in zip(k1_vals, k2_vals):
            counts[j, k1, k2] += 1

    data = []
    for j in range(count_theta):
        for k1 in range(9):
            for k2 in range(9):
                if counts[j, k1, k2] > 0:
                    data.append((j, k1, k2, counts[j, k1, k2]))
    return data