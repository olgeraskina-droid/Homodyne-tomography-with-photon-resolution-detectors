import numpy as np
from math import comb
from config import d, dim_s, theta_array
from utils import ubs, density_matrix_coherent_lo

_compute_pk_cache = {}
_precompute_cache = {}

def calculate_q_with_loss(t, loss):
    """Compute effective detector click probabilities q for a given T and loss."""
    q = np.zeros(8)
    p_det = [1 - loss[i] for i in range(len(loss))]
    q[0] = t[0] * t[1] * t[3] * p_det[0]
    q[1] = t[0] * t[1] * (1 - t[3]) * p_det[1]
    q[2] = t[0] * (1 - t[1]) * t[4] * p_det[2]
    q[3] = t[0] * (1 - t[1]) * (1 - t[4]) * p_det[3]
    q[4] = (1 - t[0]) * t[2] * t[5] * p_det[4]
    q[5] = (1 - t[0]) * t[2] * (1 - t[5]) * p_det[5]
    q[6] = (1 - t[0]) * (1 - t[2]) * t[6] * p_det[6]
    q[7] = (1 - t[0]) * (1 - t[2]) * (1 - t[6]) * p_det[7]
    return q

def compute_pk_given_n(q, n, max_k=8):
    """Probability P(k|n) for tree detector."""
    key = (tuple(q), n, max_k)
    if key in _compute_pk_cache:
        return _compute_pk_cache[key]

    num_det = 8
    total_q = np.sum(q)
    dp = np.zeros((num_det + 1, n + 1, max_k + 1))
    dp[0, 0, 0] = 1.0
    cum = 0.0
    for i in range(num_det):
        if 1 - cum <= 0:
            p_cond = 0.0
        else:
            p_cond = q[i] / (1 - cum)
        for m in range(n + 1):
            rem = n - m
            if rem < 0:
                continue
            binomial_probs = np.array([
                comb(rem, x) * (p_cond ** x) * ((1 - p_cond) ** (rem - x))
                for x in range(rem + 1)
            ])
            for k in range(max_k + 1):
                prob = dp[i, m, k]
                if prob == 0:
                    continue
                for x in range(rem + 1):
                    new_m = m + x
                    new_k = k + (1 if x > 0 else 0)
                    if new_k > max_k:
                        continue
                    dp[i + 1, new_m, new_k] += prob * binomial_probs[x]
        cum += q[i]
    result = np.zeros(max_k + 1)
    for m in range(n + 1):
        for k in range(max_k + 1):
            result[k] += dp[num_det, m, k]
    result /= result.sum()
    _compute_pk_cache[key] = result
    return result

def precompute_tensors(alpha_lo_theory, r, q1_known, q2_known):
    """Precompute POVM tensors for all theta and k1,k2 combinations."""
    key = (tuple(theta_array), alpha_lo_theory, r, tuple(q1_known), tuple(q2_known))
    if key in _precompute_cache:
        return _precompute_cache[key]

    u = ubs(d, r)
    p1_probs = np.zeros((d, 9))
    p2_probs = np.zeros((d, 9))
    for n in range(d):
        p1_probs[n] = compute_pk_given_n(q1_known, n, 8)
        p2_probs[n] = compute_pk_given_n(q2_known, n, 8)

    basis_s_full = np.zeros((dim_s, d), dtype=complex)
    for i in range(dim_s):
        basis_s_full[i, i] = 1.0

    tensors_w = []

    for theta in theta_array:
        alpha = alpha_lo_theory * np.exp(1j * theta)
        lo_vec = np.zeros(d, dtype=complex)
        lo_vec[0] = np.exp(-abs(alpha)**2 / 2)
        for n in range(1, d):
            lo_vec[n] = lo_vec[n-1] * alpha / np.sqrt(n)

        psi_in = np.zeros((d*d, dim_s), dtype=complex)
        for i in range(dim_s):
            psi_in[:, i] = np.kron(lo_vec, basis_s_full[i])
        psi_out = u @ psi_in

        m_list = []
        for n1 in range(d):
            for n2 in range(d):
                idx = n1 * d + n2
                row = psi_out[idx, :]
                m = np.outer(row.conj(), row)
                m_list.append(m)

        w = np.zeros((dim_s, dim_s, 9, 9), dtype=complex)
        idx_n = 0
        for n1 in range(d):
            for n2 in range(d):
                m = m_list[idx_n]
                idx_n += 1
                for k1 in range(9):
                    p1 = p1_probs[n1, k1]
                    if p1 == 0:
                        continue
                    for k2 in range(9):
                        p2 = p2_probs[n2, k2]
                        if p2 == 0:
                            continue
                        w[:, :, k1, k2] += m * p1 * p2

        for k1 in range(9):
            for k2 in range(9):
                w[:, :, k1, k2] = (w[:, :, k1, k2] + w[:, :, k1, k2].conj().T) / 2

        tensors_w.append(w)

    _precompute_cache[key] = tensors_w
    return tensors_w