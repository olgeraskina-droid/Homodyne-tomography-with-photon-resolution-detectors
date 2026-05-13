import numpy as np
import math
import warnings
from itertools import product
from scipy import linalg
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar, fsolve
from qutip import Qobj, fidelity as qutip_fidelity
import matplotlib.pyplot as plt
import random
import os
import scipy.linalg
from math import pi
import sys
import inspect
from functools import lru_cache
from scipy.optimize import basinhopping
from types import MethodType
from scipy.linalg import sqrtm
from scipy.optimize import OptimizeResult
import sys
import atexit
import os
from math import comb
from functools import lru_cache
import scipy.linalg as la
from scipy.optimize import fsolve

_ubs_cache = {}


def ubs(dimension, r):
    key = (dimension, r)
    if key in _ubs_cache:
        return _ubs_cache[key]
    a = np.zeros((dimension, dimension), dtype=np.complex128)
    for i in range(0, dimension - 1):
        a[i][i + 1] = np.sqrt(i + 1)
    a_t = a.conj().T
    aa = np.kron(a_t, a) - np.kron(a, a_t)
    u = linalg.expm((np.pi / 4 * r) * aa)
    _ubs_cache[key] = u
    return u


def density_matrix_coherent_lo(n, alpha_lo, theta):
    alpha_complex = alpha_lo * np.exp(1j * theta)
    norm_factor = np.exp(-alpha_lo ** 2)
    sqrt_arr = np.sqrt(np.arange(n))
    sqrt_arr[0] = 1.0
    c_m = np.ones(n, dtype=complex)
    c_n = np.ones(n, dtype=complex)
    for i in range(1, n):
        c_m[i] = c_m[i-1] * alpha_complex / sqrt_arr[i]
        c_n[i] = c_n[i-1] * np.conj(alpha_complex) / sqrt_arr[i]
    ro = norm_factor * np.outer(c_m, c_n)
    return ro


def create_random_density_matrix(dim):
    g = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = g @ g.conj().T
    type_str = f'mixed with purity {np.trace(rho @ rho).real:.6f}'
    return rho/np.trace(rho), type_str


def density_matrix_coherent(n, alpha_s, theta):
    alpha_complex = alpha_s * np.exp(1j * theta)
    norm_factor = np.exp(-alpha_s ** 2)
    sqrt_arr = np.sqrt(np.arange(n))
    sqrt_arr[0] = 1.0
    c_m = np.ones(n, dtype=complex)
    c_n = np.ones(n, dtype=complex)
    for i in range(1, n):
        c_m[i] = c_m[i-1] * alpha_complex / sqrt_arr[i]
        c_n[i] = c_n[i-1] * np.conj(alpha_complex) / sqrt_arr[i]

    ro = norm_factor * np.outer(c_m, c_n)
    ro = ro / np.trace(ro)
    type_str = 'coherent; with alpha_s = ' + str(alpha_s)
    return ro, type_str


def fock_density_matrix(n: int, N: int) -> np.ndarray:
    if n >= N:
        raise ValueError(f"n = {n} must be less than dimension N = {N}")
    rho = np.zeros((N, N), dtype=complex)
    rho[n, n] = 1.0
    type_str = 'Fock; with n = ' + str(n)
    return rho, type_str


def create_fock_comb(dim: int):

    coefficients = np.random.uniform(0, 1, size=dim)

    norm = np.sqrt(np.sum(coefficients ** 2))
    if norm == 0:
        coefficients = np.zeros(dim)
        coefficients[0] = 1.0
    else:
        coefficients = coefficients / norm
    psi = np.zeros(dim, dtype=complex)
    for i in range(dim):
        psi[i] = coefficients[i]

    psi_col = psi.reshape(-1, 1)
    rho = psi_col @ psi_col.conj().T

    type_str = 'fock with' + str(psi)
    return rho, type_str

    min_purity = 1.0 / dim
    max_purity = 1.0

    if purity < min_purity - 1e-10 or purity > max_purity + 1e-10:
        raise ValueError(f"pure must be in [{min_purity:.3f}, 1], get {purity}")

    if abs(purity - min_purity) < 1e-10:
        rho = np.eye(d) / d
        return rho, f'max mix with purity {purity:.3f}'

    if abs(purity - 1.0) < 1e-10:
        # Random vector
        psi = np.random.randn(d) + 1j * np.random.randn(d)
        psi = psi / np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        return rho, f'pure state with purity {purity:.3f}'

    # Case 3: Mixed state with given purity

    # Method 1: Use parameterization with one distinguished eigenvalue
    # Eigenvalues: [p, (1-p)/(d-1), (1-p)/(d-1), ...]
    # Then purity = p² + (d-1)*((1-p)/(d-1))² = p² + (1-p)²/(d-1)

    # Solve equation for p
    def purity_eq(p):
        return p**2 + (1-p)**2/(d-1) - purity

    p0 = (purity - min_purity)/(1 - min_purity) * (1 - 1/d) + 1/d

    try:
        p_solution = fsolve(purity_eq, p0)[0]
        # Constrain p in range [1/d, 1]
        p_solution = max(1/d, min(1.0, p_solution))
    except:
        # If solution not found, use approximation
        p_solution = p0

    # Form eigenvalues
    lambdas = np.ones(d) * (1 - p_solution) / (d - 1)
    lambdas[0] = p_solution

    # Shuffle eigenvalues (so there is no order)
    np.random.shuffle(lambdas)

    # Method 2: More general method - use random Dirichlet distribution
    # and scale it to achieve desired purity

    def generate_from_dirichlet(alpha=1.0):
        """Generate random distribution using Dirichlet distribution"""
        # Generate random vector from Dirichlet distribution
        dirichlet_vec = np.random.dirichlet([alpha] * d)

        # Optimize to achieve desired purity
        def objective(scale):
            scaled = dirichlet_vec ** scale
            scaled = scaled / np.sum(scaled)
            return np.sum(scaled ** 2) - purity

        try:
            scale_sol = fsolve(objective, 1.0)[0]
            scaled = dirichlet_vec ** scale_sol
            scaled = scaled / np.sum(scaled)
            return scaled
        except:
            return dirichlet_vec

    lambdas1 = lambdas.copy()
    purity1 = np.sum(lambdas1 ** 2)
    lambdas2 = generate_from_dirichlet(alpha=0.5)
    purity2 = np.sum(lambdas2 ** 2)
    if abs(purity2 - purity) < abs(purity1 - purity):
        lambdas = lambdas2
    lambdas = np.maximum(lambdas, 0)
    lambdas = lambdas / np.sum(lambdas)
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, R = la.qr(Z)
    U = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    rho = U @ np.diag(lambdas) @ U.conj().T
    rho = (rho + rho.conj().T) / 2
    rho = rho / np.trace(rho)      # Normalize trace

    # Check and fix negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(rho)
    if np.any(eigvals < -1e-10):
        eigvals = np.maximum(eigvals, 0)
        eigvals = eigvals / np.sum(eigvals)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    # Final check
    calc_purity = np.trace(rho @ rho).real
    if abs(calc_purity - purity) > 0.1:
        print(f"⚠️  Warning: obtained purity {calc_purity:.3f} differs from target {purity:.3f}")

    return rho, f'mixed with purity {calc_purity:.3f} (target {purity:.3f})'


def create_ro_s(str_type, n, alpha_s):
    if str_type == 'c':
        ro_s_0, type_str = density_matrix_coherent(dim_s, alpha_s, 0)
    elif str_type == 'f':
        ro_s_0, type_str = fock_density_matrix(n, dim_s)
    elif str_type == 'fc':
        ro_s_0, type_str = create_fock_comb(dim_s)
    else:
        ro_s_0, type_str = create_random_density_matrix(dim_s)
    return ro_s_0, type_str


def is_density_matrix(matrix, tol=1e-10):
    if not np.allclose(matrix, matrix.conj().T, atol=tol):
        print("Not Hermitian")
        return False

    trace = np.trace(matrix)
    if not np.isclose(trace, 1.0, atol=tol):
        print(f"(trace = {trace:.10f})")
        return False

    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(eigenvalues >= -tol):
        print(f"❌ Negative eigenvalues: {eigenvalues}")
        return False

    return True


def params_to_rho(p, dim):
    t = np.zeros((dim, dim), dtype=complex)
    idx = 0
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                t[i, i] = p[idx]
                idx += 1
            else:
                t[i, j] = p[idx] + 1j * p[idx+1]
                idx += 2
    rho = t @ t.conj().T
    return rho / np.trace(rho)


def calculate_q_with_loss(t, loss):
    q = np.zeros(8)
    p_det = []
    for i in range(len(loss)):
        p_det.append(1 - loss[i])
    q[0] = t[0] * t[1] * t[3] * p_det[0]
    q[1] = t[0] * t[1] * (1 - t[3]) * p_det[1]
    q[2] = t[0] * (1 - t[1]) * t[4] * p_det[2]
    q[3] = t[0] * (1 - t[1]) * (1 - t[4]) * p_det[3]
    q[4] = (1 - t[0]) * t[2] * t[5] * p_det[4]
    q[5] = (1 - t[0]) * t[2] * (1 - t[5]) * p_det[5]
    q[6] = (1 - t[0]) * (1 - t[2]) * t[6] * p_det[6]
    q[7] = (1 - t[0]) * (1 - t[2]) * (1 - t[6]) * p_det[7]
    return q


_compute_pk_cache = {}


def compute_pk_given_n(q, n, max_k=8):
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


_precompute_cache = {}


def precompute_tensors(alpha_lo_theory, r, q1_known, q2_known):
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


def neg_log_likelihood_pvm(p, data, tensors_w):
    rho = params_to_rho(p, dim_s)
    logL = 0.0
    eps = 1e-12
    for theta_idx, k1, k2, cnt in data:
        W = tensors_w[theta_idx][:, :, k1, k2]
        prob = np.trace(rho @ W).real
        prob = max(prob, eps)
        logL += cnt * np.log(prob)
    return -logL


def create_data(ro_s_0, alpha_lo_real, R,
                t1, t2,
                loss1_real, loss2_real):

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

        def simulate_tree_fast(n_arr, q_eff):
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


def rrhor_pvm(data, dim_s, tensors_w, max_iter=150, tol=1e-12):
    rho = np.eye(dim_s, dtype=complex) / dim_s
    old_logL = -np.inf
    for it in range(max_iter):
        update = np.zeros((dim_s, dim_s), dtype=complex)
        for theta_idx, k1, k2, cnt in data:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob_total = np.trace(rho @ W).real
            if prob_total < 1e-12:
                continue
            weight = cnt / prob_total
            update += weight * W
        rho_new = rho @ update
        eigvals, eigvecs = np.linalg.eigh(rho_new)
        eigvals = np.maximum(eigvals, 0)
        eigvals /= eigvals.sum()
        rho_new = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        logL = 0.0
        for theta_idx, k1, k2, cnt in data:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob_total = np.trace(rho_new @ W).real
            logL += cnt * np.log(max(prob_total, 1e-12))
        if abs(logL - old_logL) < tol:
            break
        old_logL = logL
        rho = rho_new
    return rho


def rho_to_params(rho, dim_s):

    rho_reg = rho + 1e-12 * np.eye(dim_s)
    try:
        L = np.linalg.cholesky(rho_reg)
    except np.linalg.LinAlgError:
        # If decomposition failed, use matrix square root
        from scipy.linalg import sqrtm
        L = sqrtm(rho_reg)
        # L may not be triangular, but params_to_rho expects triangular?
        # In that case it's better to return a random p0, but we hope rho is sufficiently positive.
        # Alternative: use eigenvectors to build a unitary transform, but that's complicated.
        # For simplicity: return parameters corresponding to max mixed state.
        p0 = np.zeros(dim_s**2)
        p0[:dim_s] = np.log(np.ones(dim_s) / dim_s)
        p0[dim_s:] = np.random.normal(scale=0.1, size=dim_s*(dim_s-1))
        return p0
    trace = np.trace(L @ L.conj().T).real
    L /= np.sqrt(trace)
    p = []
    idx = 0
    for i in range(dim_s):
        for j in range(i+1):
            if i == j:
                p.append(L[i, i].real)
                idx += 1
            else:
                p.append(L[i, j].real)
                p.append(L[i, j].imag)
                idx += 2
    return np.array(p)


def adam_minimize(loss_fn, p0, args, max_iter=500, lr=0.01):
    p = p0.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    best_p = p.copy()
    best_loss = np.inf

    data_full = args[0]  # X_theta

    for t in range(1, max_iter+1):
        # Mini-batch: random 50% of data
        batch_size = max(1, len(data_full) // 2)
        idx = np.random.choice(len(data_full), batch_size, replace=False)
        data_batch = [data_full[i] for i in idx]

        # Numerical gradient (can be replaced with autograd)
        grad = np.zeros_like(p)
        eps_grad = 1e-8
        loss0 = loss_fn(p, data_batch, *args[1:])

        for i in range(len(p)):
            p_step = p.copy()
            p_step[i] += eps_grad
            loss1 = loss_fn(p_step, data_batch, *args[1:])
            grad[i] = (loss1 - loss0) / eps_grad

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)

        # Full evaluation on all data
        if t % 50 == 0:
            loss_full = loss_fn(p, data_full, *args[1:])
            if loss_full < best_loss:
                best_loss = loss_full
                best_p = p.copy()

    return OptimizeResult(x=best_p, fun=best_loss, success=True)


def Fidelity_R_with_tree_povm(R, ro_s_0, alpha_lo_theory, alpha_LO_real,
                              t1, t2, t_know,
                              loss1_real, loss2_real, loss_know):
    data_pvm = create_data(ro_s_0, alpha_LO_real,
                           R, t1, t2,
                           loss1_real, loss2_real)
    q1_known = calculate_q_with_loss(t_know, loss_know)
    q2_known = q1_known
    tensors_w = precompute_tensors(alpha_lo_theory, R,
                                   q1_known, q2_known)

    total_counts = sum(cnt for _,_,_,cnt in data_pvm)
    gamma = 0 * total_counts / dim_s

    p0_RroR = rho_to_params(rrhor_pvm(data_pvm, dim_s, tensors_w, max_iter=100), dim_s)
    print('init. rho')
    print(np.round(ro_s_0, 4))

    f_best = 0
    fun_best = np.inf
    count_start = 0
    while count_start < 1:
        if count_start == 0:
            '''minimizer_kwargs = {
              'method': 'L-BFGS-B',
              'args': (data_pvm, tensors_w),
              'options': {'maxiter': 1000, 'ftol': 1e-7}
            }
            result = basinhopping(neg_log_likelihood_pvm, p0_RroR,
                                  minimizer_kwargs=minimizer_kwargs,
                                  niter=150, stepsize=0.5)'''
            result = minimize(neg_log_likelihood_pvm, p0_RroR,
                              args=(data_pvm, tensors_w),
                              method='L-BFGS-B')

        elif count_start == 1:
            result = adam_minimize(neg_log_likelihood_pvm, p0_RroR,
                          args=(data_pvm, dim_s, tensors_w, gamma))


            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'args': (data_pvm, dim_s, tensors_w, gamma),
                'options': {'maxiter': 1000, 'ftol': 1e-10}
            }
            result = basinhopping(neg_log_likelihood_pvm, result.x,
                                  minimizer_kwargs=minimizer_kwargs,
                                  niter=150, stepsize=0.5)

        elif count_start == 2:
            p0 = np.random.randn(dim_s**2) * 0.6
            result = adam_minimize(neg_log_likelihood_pvm, p0,
                                   args=(data_pvm, dim_s, tensors_w, gamma))
            result = minimize(neg_log_likelihood_pvm, result.x,
                              args=(data_pvm, dim_s, tensors_w, gamma),
                              method='L-BFGS-B')

        else:
            p0 = np.random.randn(dim_s**2) * 0.8
            minimizer_kwargs = {
              'method': 'L-BFGS-B',
              'args': (data_pvm, dim_s, tensors_w, gamma),
              'options': {'maxiter': 1000, 'ftol': 1e-7}
            }
            result = basinhopping(neg_log_likelihood_pvm, p0,
                                  minimizer_kwargs=minimizer_kwargs,
                                  niter=100, stepsize=0.5)
        rec_rho = params_to_rho(result.x, dim_s)
        f = float(qutip_fidelity(Qobj(ro_s_0), Qobj(rec_rho / np.trace(rec_rho))))
        print(' ')
        print('rec rho')
        print(np.round(rec_rho, 4))
        if f > f_best:
          f_best = f
          recon_rho = rec_rho
        if result.fun < fun_best:
          best_res_fun = result.fun
          f_best_fun = f
        count_start += 1
        #print('count_start = ' + str(count_start) + ', f = ' + str(f) + ', fun = ' + str(result.fun))


    '''print('Result')

    print('f_best = ' + str(f_best))'''
    print('f_best_fun = ' + str(f_best_fun))
    if np.abs(f_best - f_best_fun) > 0.015:
      print('DID NOT MATCH!!!!!!!!!!!!!!!!' + ' best fun fidelity: ' + str(f_best_fun))

    return f_best


# CONST


count_theta = 30
iterations = 1000
d = 10
dim_s = 3

#all_err = 0.1
loss_KNOW = [0.55] * 8
T_know = [0.5] * 7
alpha_LO_theory = 2
theta_array = np.linspace(0, 2 * np.pi, count_theta)
#######################################

'''T1_real = [t_know[0] / (1 - all_err)] * 7
T2_real = [t_know[0] / (1 - all_err)] * 7
loss1 = [loss_know[0] / (1 - all_err)] * 8
loss2 = [loss_know[0] / (1 - all_err)] * 8'''
R = 1

def create_pic_f_t(start_T_sr, stop_T_sr, count_T, count_exp,
                   alpha_lo_theory=alpha_LO_theory,
                   loss_know=loss_KNOW, t_know=T_know, r=R):
    if t_know is None:
        t_know = T_know
    fidelity_err = np.zeros((count_exp, count_T), dtype=float)
    import random

    step = (stop_T_sr - start_T_sr) / count_T
    arrays_data = []

    for i in range(count_T):
        segment_start = start_T_sr + i * step
        segment_end = start_T_sr + (i + 1) * step
        T = [round(random.uniform(segment_start, segment_end), 3) for _ in range(7)]
        mean_value = sum(T) / len(T)
        arrays_data.append((T, mean_value))

    arrays_data.sort(key=lambda x: x[1])
    T1_matrix = [T for T, mean in arrays_data]


    #All T same and equal to the mean
    target_means = np.linspace(start_T_sr, stop_T_sr, count_T)
    T1_matrix = []
    for mean in target_means:

        T = [round(mean, 3) for _ in range(7)]
        T1_matrix.append(T)
    i = 0
    while i < count_exp:
        j = 0
        while j < count_T:
            ro_s_0, type_str = create_ro_s('hk', n=0, alpha_s=0)
            n2 = ro_s_0[2][2]
            if n2 > 0.17: continue
            fidelity_err[i][j] = 1 - Fidelity_R_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                                               alpha_lo_real=alpha_LO_theory,
                                                               r=R, T1_real=T1_matrix[j], T2_real=T1_matrix[j], t_know=T_know,
                                                               loss1_real=loss_KNOW, loss2_real=loss_KNOW, loss_know=loss_know
                                                               )
            progress = ((j + 1) / (count_T * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)
            j += 1
        i += 1

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)
    T_otn_err = []
    for i in range(count_T):
        T_otn_err.append(100 * np.abs(1 - sum(T1_matrix[i]) / len(T1_matrix[i]) / t_know[0]))

    plt.figure(figsize=(10, 6))
    plt.errorbar(T_otn_err, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('T error (%)', fontsize=10)
    plt.ylabel('1 - Fidelity', fontsize=12)
    plt.xticks(rotation=45)
    #plt.yscale('log')
    #plt.xticks(np.arange(0, count_T, 2))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"f_t.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPic save in: {filepath}")
    plt.close()


def create_pic_f_detect_loss(start_loss, stop_loss, count_loss, count_exp,
                             alpha_lo_theory=alpha_LO_theory,
                             loss_know=loss_KNOW, t_know=T_know, r=R):

    fidelity_err = np.zeros((count_exp, count_loss), dtype=float) # fidelity error matrix
    import random

    step = (stop_loss - start_loss) / count_loss
    arrays_data = []

    for i in range(count_loss):
        segment_start = start_loss + i * step
        segment_end = start_loss + (i + 1) * step
        loss = [round(random.uniform(segment_start, segment_end), 3) for _ in range(8)]
        mean_value = sum(loss) / len(loss)
        arrays_data.append((loss, mean_value))

    arrays_data.sort(key=lambda x: x[1])
    loss1_matrix = [loss for loss, mean in arrays_data]
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Calculation of fidelity errors
    i = 0
    while i < count_exp:
        j = 0
        while j < count_loss:
            ro_s_0, type_str = create_ro_s('hk', n=0, alpha_s=0)
            n2 = ro_s_0[2][2]
            if n2 > 0.17: continue

            fidelity_err[i][j] = 1 - Fidelity_r_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                                        alpha_lo_real=alpha_LO_theory,
                                                        r=R, T1_real=T_know, T2_real=T_know, t_know=T_know,
                                                        loss1_real=loss1_matrix[j], loss2_real=loss1_matrix[j],
                                                        loss_know=loss_know
                                                        )
            progress = ((j + 1) / (count_loss * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)
            j += 1
        i += 1

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)
    loss_otn_err = []
    for i in range(count_loss):
        loss_otn_err.append(100 * np.abs(1 - (sum(loss1_matrix[i])) / (len(loss1_matrix[i])) / loss_know[0]))
    plt.figure(figsize=(10, 6))
    plt.errorbar(loss_otn_err, means * 100, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('loss err (%)', fontsize=10)
    plt.ylabel('1 - Fidelity', fontsize=12)
    plt.xticks(rotation=45)
    #plt.yscale('log')
    #plt.xticks(np.arange(0, count_T, 2))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"loss_alpha_known_loss_{loss_know[1]}.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def create_pic_f_alpha(start_a, stop_a, count_a, count_exp,
                       alpha_lo_theory=alpha_LO_theory,
                       loss_know=loss_KNOW, t_know=T_know, r=R):

    fidelity_err = np.zeros((count_exp, count_a), dtype=float) # fidelity error matrix
    alpha_arr = np.linspace(start_a, stop_a, count_a)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Calculation of fidelity errors
    i = 0
    while i < count_exp:
        j = 0
        while j < count_a:
            ro_s_0, type_str = create_ro_s('hk', n=0, alpha_s=0)
            n2 = ro_s_0[2][2]
            if n2 > 0.17: continue

            fidelity_err[i][j] = 1 - Fidelity_r_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                                        alpha_lo_real=alpha_arr[j],
                                                        r=R, T1_real=T_know, T2_real=T_know, t_know=T_know,
                                                        loss1_real=loss_KNOW, loss2_real=loss_KNOW,
                                                        loss_know=loss_know
                                                        )
            progress = ((j + 1) / (count_a * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)
            j += 1
        i += 1

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(100 * np.abs(1 - alpha_arr/alpha_lo_theory), means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('alpha LO error (%)', fontsize=10)
    plt.ylabel('1 - Fidelity', fontsize=12)
    plt.xticks(rotation=45)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'f_alpha.png'
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def create_pic_f_error(start_err, stop_err, count_err, count_exp,
                       alpha_lo_theory=alpha_LO_theory,
                       loss_know=loss_KNOW, t_know=T_know, r=R):
    fidelity_err = np.zeros((count_exp, count_err), dtype=float)
    error_arr = np.linspace(start_err, stop_err, count_err)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #Calculation of fidelity errors
    i = 0
    while i < count_exp:
        j = 0
        while j < count_err:
            error_all = error_arr[j]
            t1 = [t_know[0] / (1 - error_all)] * 7
            loss1 = [loss_know[0] / (1 - error_all)] * 8
            t2 = t1
            loss2 = loss1
            ro_s_0, type_str = create_ro_s('hk', n=0, alpha_s=0)
            n2 = ro_s_0[2][2]
            if n2 > 0.17: continue
            alpha_lo_real = reconstructed_alfa_pvm(r=1, alpha_lo_real=2, t1=t1, t2=t2, t_know=T_know,
                                                    loss1=loss1, loss2=loss2, loss_know=loss_know)
            fidelity_err[i][j] = 1 - Fidelity_r_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                                               alpha_lo_real=alpha_lo_real,
                                                               r=R, T1_real=t1, T2_real=t1, t_know=T_know,
                                                               loss1_real=loss1, loss2_real=loss1,
                                                               loss_know=loss_know
                                                               )
            progress = ((j + 1) / (count_err * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)
            j += 1
        i += 1

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(100 * error_arr, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('error of T loss (%)', fontsize=10)
    plt.ylabel('1 - Fidelity', fontsize=12)
    plt.xticks(rotation=45)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"f_error_T_loss.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def create_pic_reconstruct_alpha_t_loss(start_err, stop_err, count_err, count_exp,
                                        alpha_lo_theory=alpha_LO_theory,
                                        loss_know=loss_KNOW, t_know=T_know, r=R
                                        ):
    error_arr = np.linspace(start_err, stop_err, count_err)

    alpha_err_matrix = np.zeros((count_exp, count_err), dtype=float)
    for j, error in enumerate(error_arr):
        T_real = [t_know[0] / (1 - error)] * 7
        loss_real = [loss_know[0] / (1 - error)] * 8
        for i in range(count_exp):
            alpha_real = reconstructed_alfa_pvm(r, alpha_lo_theory,
                                                 t1=T_real, t2=T_real, t_know=T_know,
                                                 loss1=loss_real, loss2=loss_real, loss_know=loss_know
                                                 )
            alpha_err_matrix[i][j] = 100 * np.abs(1 - alpha_real/alpha_lo_theory)


    means = np.mean(alpha_err_matrix, axis=0)
    std_errors = np.std(alpha_err_matrix, axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(error_arr * 100, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('error of T and loss (%)', fontsize=10)
    plt.ylabel('alpha LO error (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"a_error_T_loss.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def create_pic_reconstruct_alpha_t(start_err, stop_err, count_err, count_exp,
                                   alpha_lo_theory=alpha_LO_theory,
                                   loss_know=loss_KNOW, t_know=T_know, r=R
                                   ):
    error_arr = np.linspace(start_err, stop_err, count_err)
    i = 0
    alpha_err_matrix = np.zeros((count_exp, count_err), dtype=float)
    for j in range(count_err):
        error = error_arr[j]
        T_real = [t_know[0] / (1 - error)] * 7

        for i in range(count_exp):
            alpha_real = reconstructed_alfa_pvm(r, alpha_lo_theory,
                                                 t1=T_real, t2=T_real, t_know=T_know,
                                                 loss1=loss_KNOW, loss2=loss_KNOW, loss_know=loss_know
                                                 )
            alpha_err_matrix[i][j] = 100 * np.abs(1 - alpha_real/alpha_lo_theory)


    means = np.mean(alpha_err_matrix, axis=0)
    std_errors = np.std(alpha_err_matrix, axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(error_arr * 100, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('error of T (%)', fontsize=10)
    plt.ylabel('alpha LO error (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"a_error_T.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def create_pic_reconstruct_alpha_loss(start_err, stop_err, count_err, count_exp,
                                      alpha_lo_theory=alpha_LO_theory,
                                      loss_know=loss_KNOW, t_know=T_know, r=R
                                      ):
    error_arr = np.linspace(start_err, stop_err, count_err)
    print(error_arr)
    alpha_err_matrix = np.zeros((count_exp, count_err), dtype=float)
    for j in range(count_err):
        error = error_arr[j]
        loss_real = [loss_know[0] / (1 - error)] * 8
        for i in range(count_exp):
            alpha_real = reconstructed_alfa_pvm(r, alpha_lo_theory,
                                                 t1=T_know, t2=T_know, t_know=T_know,
                                                 loss1=loss_real, loss2=loss_real, loss_know=loss_know
                                                 )
            alpha_err_matrix[i][j] = 100 * np.abs(1 - alpha_real/alpha_lo_theory)

    means = np.mean(alpha_err_matrix, axis=0)
    std_errors = np.std(alpha_err_matrix, axis=0, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.errorbar(error_arr * 100, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2)
    plt.xlabel('error of loss (%)', fontsize=10)
    plt.ylabel('alpha LO error (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"a_error_loss.png"
    filepath = os.path.join(script_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {filepath}")
    plt.close()


def reconstructed_alfa_pvm(r, alpha_lo_real,
                           t1, t2, t_know,
                           loss1, loss2, loss_know
                           ):

    ro_s_0, type_str = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(
        ro_s_0, alpha_lo_real, r,
        t1, t2,
        loss1, loss2
    )
    q_known = calculate_q_with_loss(t_know, loss_know)

    def neg_log_lik(alpha):
        # Recalculate POVM elements for this alpha
        tensors_w = precompute_tensors(alpha, r, q_known, q_known)
        logL = 0.0
        eps = 1e-12
        for theta_idx, k1, k2, cnt in data_pvm:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob = np.trace(ro_s_0 @ W).real
            prob = max(prob, eps)
            logL += cnt * np.log(prob)
        return -logL

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(neg_log_lik, bounds=(0.1, 10), method='bounded')
    if result.success:
        return result.x
    else:
        print(f"Alpha optimization did not converge, returning 1.0")
        return 1.0


def reconstruct_param_a_loss(r, alpha_lo_real,
                             t1, t2, t_know,
                             loss1, loss2):
    ro_s_0, type_str = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(
        ro_s_0, alpha_lo_real, r,
        t1, t2,
        loss1, loss2
    )

    def calibration_loss(params, data, rr):
        alpha = params[0]
        t = t_know
        loss = params[1:9]
        q_eff = calculate_q_with_loss(t, loss)
        tensors_w = precompute_tensors(alpha, rr, q_eff, q_eff)
        logs = 0.0
        eps = 1e-12
        for theta_idx, k1, k2, cnt in data:
            w = tensors_w[theta_idx][:, :, k1, k2]
            prob = w[0, 0].real
            prob = max(prob, eps)
            logs += cnt * np.log(prob)
        return -logs
    bounds = [(1.9, 3)] + [(0.5, 0.7)] * 8
    p0 = [2.0] + [0.55] * 8
    result = minimize(calibration_loss, p0, args=(data_pvm, r),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    estimated_params = result.x
    return estimated_params


def reconstruct_loss(r, alpha_lo_theory, alpha_lo_real,
                     t1, t2, t_know,
                     loss1, loss2):
    ro_s_0, type_str = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(
        ro_s_0, alpha_lo_real, r,
        t1, t2,
        loss1, loss2
    )
    def calibration_loss(params, data_pvm, r):
        alpha = alpha_lo_theory
        T = t_know
        loss = params[0:8]
        q_eff = calculate_q_with_loss(T, loss)
        tensors_w = precompute_tensors(alpha, r, q_eff, q_eff)
        logL = 0.0
        eps = 1e-12
        for theta_idx, k1, k2, cnt in data_pvm:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob = W[0, 0].real
            prob = max(prob, eps)
            logL += cnt * np.log(prob)
        return -logL

    result = minimize(calibration_loss, [0.55] * 8, args=(data_pvm, r),
                      method='L-BFGS-B', bounds=[(0.45, 1)] * 8, options={'maxiter': 1000})
    estimated_params = result.x
    return estimated_params


def reconstruct_param(r, alpha_lo_theory,
                      t1, t2,
                      loss1, loss2):
    ro_s_0, type_str = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(
        ro_s_0, alpha_lo_theory, r,
        t1, t2,
        loss1, loss2
    )
    def calibration_loss(params, data_pvm, r):
        alpha = params[0]
        T = params[1:8]
        loss = params[8:16]
        q_eff = calculate_q_with_loss(T, loss)
        tensors_w = precompute_tensors(alpha, r, q_eff, q_eff)
        logL = 0.0
        eps = 1e-12
        for theta_idx, k1, k2, cnt in data_pvm:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob = W[0, 0].real
            prob = max(prob, eps)
            logL += cnt * np.log(prob)
        return -logL
    bounds = [(0.1, 3)] + [(0, 1)] * 7 + [(0.1, 0.8)] * 8
    p0 = [2] + [0.5] * 7 + [0.55] * 8
    result = minimize(calibration_loss, p0, args=(data_pvm, r),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
    estimated_params = result.x
    return estimated_params

'''create_pic_f_alpha(start_a=1.7, stop_a=2.3, count_a=7, count_exp=5)

create_pic_f_t(start_T_sr=0.4, stop_T_sr=0.5, count_T=5, count_exp=5)

create_pic_f_detect_loss(start_loss=0.4, stop_loss=0.7, count_loss=7, count_exp=5)

create_pic_f_error(0, 0.1, 11, 3)

create_pic_reconstr_alpha_t_loss(0, 0.1, 11, 3)

create_pic_reconstr_alpha_loss(0, 0.1, 11, 3)

create_pic_reconstr_alpha_t(0, 0.1, 11, 3)'''


'''alpha_loss = reconstr_param_a_loss(R=R, alpha_LO_real=2,
                                  t1=T1_real, t2=T2_real, t_know=T_know,
                                  loss1=loss1_true, loss2=loss1_true)

#alpra_rec = alpha_loss[0]
loss_rec = alpha_loss[1:9]
print(alpra_rec, loss_rec)'''
'''F_arr = []
print(loss1_true)
loss_rec = reconstruct)_loss(R=R, alpha_lo_theory=2, alpha_LO_real=1.95,
                   t1=T1_real, t2=T2_real, t_know=T_know,
                   loss1=loss1_true, loss2=loss1_true)
print(loss_rec)
c = 0
while c < 10:
    ro_s_0, type = create_ro_s('afaf', 0, 0)
    n2 = ro_s_0[2][2]
    if n2 > 0.2:
        continue
    f = Fidelity_R_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=2,
                                  alpha_LO_real=1.95,
                                  R=R, T1_real=T1_real, T2_real=T2_real, t_know=T_know,
                                  loss1_real=loss1_true, loss2_real=loss1_true,
                                  loss_know=loss_rec
                                  )
    F_arr.append(f)
    c += 1
print(np.mean(F_arr), np.std(F_arr))'''

'''rec = reconstr_param(R=R, alpha_lo_theory=alpha_LO_theory,
                            t1=T1_real, t2=T2_real,
                            loss1=loss1, loss2=loss2)

alpra_rec = rec[0]
t_rec = rec[1:8]
loss_rec = rec[8:16]

F_arr = []
print(t_rec)
c = 0
while c < 10:
    ro_s_0, type = create_ro_s('afaf', 0, 0, 0.5)
    n2 = ro_s_0[2][2]
    if n2 > 0.2:
        continue
    print(c)
    f = Fidelity_R_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                  alpha_LO_real=alpra_rec,
                                  R=R, T1_real=t_rec, T2_real=t_rec, t_know=T_know,
                                  loss1_real=loss_rec, loss2_real=loss_rec,
                                  loss_know=loss_know
                                  )
    F_arr.append(f)
    c += 1
print(np.mean(F_arr), np.std(F_arr))'''


def pic_f_a_with_reconstruction_loss(start_a, stop_a, count_a, count_exp,
                                     alpha_lo_theory=alpha_LO_theory, R=R):
    loss_true = [round(random.uniform(0.5, 0.65), 3) for _ in range(8)]
    t_true = [round(random.uniform(0.42, 0.58), 3) for _ in range(7)]
    fidelity_err = np.zeros((count_exp, count_a), dtype=float)
    alpha_arr = np.linspace(start_a, stop_a, count_a)
    j = 0
    while j < count_a:
        i = 0
        loss_rec = reconstruct_loss(r=R, alpha_lo_theory=alpha_LO_theory, alpha_lo_real=alpha_arr[j],
                                    t1=t_true, t2=t_true, t_know=T_know,
                                    loss1=loss_true, loss2=loss_true
                                    )

        while i < count_exp:
            ro_s_0, type_str = create_ro_s('hk', n=0, alpha_s=0)
            if (ro_s_0[2][2] > 0.1 or ro_s_0[0][0] < 0.5):
                continue
            avg = np.sum([0, 1, 2] * np.diag(ro_s_0).real)
            print(avg, 'average')
            fidelity_err[i][j] = 1 - Fidelity_R_with_tree_povm(ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                                                               alpha_LO_real=alpha_arr[j],
                                                               R=R, t1=t_true, t2=t_true, t_know=T_know,
                                                               loss1_real=loss_true, loss2_real=loss_true,
                                                               loss_know=loss_rec
                                                               )
            progress = ((i + 1) / (count_exp * count_a) + j / count_a) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)
            i += 1
        j += 1

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)
    x_over = []
    means_over = []
    std_over = []
    x_under = []
    means_under = []
    std_under = []

    for j, alpha in enumerate(alpha_arr):
        x_val = 100 * np.abs(1 - alpha / alpha_lo_theory)
        if alpha >= alpha_lo_theory:
            x_over.append(x_val)
            means_over.append(means[j])
            std_over.append(std_errors[j])
        if alpha <= alpha_lo_theory:
            x_under.append(x_val)
            means_under.append(means[j])
            std_under.append(std_errors[j])
    #script_dir = os.path.dirname(os.path.abspath(__file__))

    def get_unique_filename(base_path):
        if not os.path.exists(base_path):
            return base_path
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        counter = 1
        while True:
            new_name = f"{name}_{counter}{ext}"
            new_path = os.path.join(directory, new_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    '''if x_over:
        base_over = os.path.join(script_dir, 'f_alpha_withrecloss_over.csv')
        over_file = get_unique_filename(base_over)
        over_data = np.column_stack((x_over, means_over, std_over))
        np.savetxt(over_file, over_data, delimiter=',',
                   header='relative_error,mean,std', comments='', fmt='%.6f')
        print(f"\nOver data saved to: {over_file}")
    else:
        print("No data for overestimation")

    if x_under:
        base_under = os.path.join(script_dir, 'f_alpha_withrecloss_under.csv')
        under_file = get_unique_filename(base_under)
        under_data = np.column_stack((x_under, means_under, std_under))
        np.savetxt(under_file, under_data, delimiter=',',
                   header='relative_error,mean,std', comments='', fmt='%.6f')
        print(f"Under data saved to: {under_file}")
    else:
        print("No data for underestimation")
    base_raw = os.path.join(script_dir, 'f_alpha_withrecloss_raw.npz')
    raw_file = get_unique_filename(base_raw)
    np.savez(raw_file, alpha_arr=alpha_arr, fidelity_err=fidelity_err)
    print(f"Raw data saved to: {raw_file}")'''

    plt.figure(figsize=(12, 7))
    plt.errorbar(x_over, means_over, yerr=std_over,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', linewidth=2,
                 color='red', label='Overestimation (α > α₀)')
    plt.errorbar(x_under, means_under, yerr=std_under,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='s', markersize=6, linestyle='-', linewidth=2,
                 color='blue', label='Underestimation (α < α₀)')

    plt.xlabel('Relative error of α LO |1 - α/α₀| (%)', fontsize=14)
    plt.ylabel('1 - Fidelity', fontsize=14)
    plt.title('Fidelity error vs α LO mismatch with reconstruction loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    base_png = os.path.join(script_dir, 'f_alpha_withrecloss.png')
    png_file = get_unique_filename(base_png)
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {png_file}")




'''if __name__ == "__main__":
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    pic_f_a_with_reconstruction_loss(1.8, 2.2, 2, 2)   # your call
    profiler.disable()
    profiler.dump_stats('profile_results.prof')
    print("Profiling completed, data saved to profile_results.prof")'''
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import minimize_scalar, minimize
from typing import Optional, Dict, Any, Union, List

# It is assumed that all previous definitions (ubs, density_matrix_coherent_lo,
# create_ro_s, Fidelity_R_with_tree_povm, reconstructed_alfa_pvm, reconstruct_loss, etc.)
# are already loaded into the environment.

def plot_fidelity_vs_mismatch_two_cases(
    start_err: float = 0.0,
    stop_err: float = 0.2,
    num_points: int = 10,
    num_experiments: int = 5,
    alpha_nom: float = None,
    loss_nom: Union[float, List[float]] = None,
    t_nom: Union[float, List[float]] = None,
    R: float = None,
    filename_base: str = "fidelity_two_cases",
    save_data: bool = True,
    load_data: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Plots two curves on one graph:
      1. Dependence of 1 - Fidelity on the relative error in alpha LO
         provided that the losses (loss) are reconstructed from the data.
      2. Dependence of 1 - Fidelity on the relative error in loss
         provided that alpha LO is reconstructed from the data.

    Parameters
    ----------
    start_err, stop_err : float
        Range of relative errors (in fractions, e.g. 0.1 for 10%).
    num_points : int
        Number of points within the range.
    num_experiments : int
        Number of random system states for averaging at each point.
    alpha_nom : float, optional
        Nominal value of alpha LO. If None, the global alpha_LO_theory is taken.
    loss_nom : float or list of 8 floats, optional
        Nominal losses. If a number, all 8 detectors have the same value.
        If None, the global loss_KNOW is taken.
    t_nom : float or list of 7 floats, optional
        Nominal transmission coefficients T. If None, the global T_know is taken.
    R : float, optional
        Squeezing parameter. If None, the global R is taken.
    filename_base : str
        Base name for saved files (without extension).
    save_data : bool
        If True, saves the results to a .npz file.
    load_data : dict, optional
        If a dictionary with previously saved data is passed (keys 'err_vals',
        'fidelity_alpha_mean', 'fidelity_alpha_std', 'fidelity_loss_mean',
        'fidelity_loss_std'), the graph is built directly from them.
    verbose : bool
        Print execution progress.
    **kwargs : dict
        Additional parameters passed to the reconstruction and fidelity calculation functions.

    Returns
    ----------
    results : dict
        Dictionary with data:
            'err_vals' : np.ndarray
            'fidelity_alpha_mean' : np.ndarray
            'fidelity_alpha_std'  : np.ndarray
            'fidelity_loss_mean'  : np.ndarray
            'fidelity_loss_std'   : np.ndarray
            'raw_data' : dict (optional) – raw fidelity values for each experiment.
    """
    # If ready-made data is passed, plot the graph and return them
    if load_data is not None:
        required_keys = ['err_vals', 'fidelity_alpha_mean', 'fidelity_alpha_std',
                         'fidelity_loss_mean', 'fidelity_loss_std']
        if not all(k in load_data for k in required_keys):
            raise ValueError("load_data must contain all keys: " + ", ".join(required_keys))
        _plot_two_cases(load_data)
        return load_data

    # Use global default values if not explicitly provided
    global alpha_LO_theory, loss_KNOW, T_know
    if alpha_nom is None:
        alpha_nom = alpha_LO_theory
    if loss_nom is None:
        loss_nom = loss_KNOW
    if t_nom is None:
        t_nom = T_know
    if R is None:
        R = 1

    # Convert loss_nom and t_nom to lists of required length
    if isinstance(loss_nom, (int, float)):
        loss_nom = [float(loss_nom)] * 8
    if isinstance(t_nom, (int, float)):
        t_nom = [float(t_nom)] * 7

    err_vals = np.linspace(start_err, stop_err, num_points)

    # Arrays for results (one value per experiment and point)
    fid_alpha = np.zeros((num_experiments, num_points))
    fid_loss  = np.zeros((num_experiments, num_points))

    total_points = num_points * num_experiments
    current = 0

    for i, err in enumerate(err_vals):
        if verbose:
            print(f"\nPoint {i+1}/{num_points}, error = {err*100:.2f}%")

        exp_count = 0
        attempts = 0
        max_attempts = num_experiments * 1000

        while exp_count < num_experiments and attempts < max_attempts:
            attempts += 1

            # Generate a random state, as in other functions
            ro_s_0, _ = create_ro_s('hk', n=0, alpha_s=0)
            # Selection condition (similar to pic_f_a_with_reconstruction_loss)
            if ro_s_0[2][2] > 0.1 or ro_s_0[0][0] < 0.5:
                continue

            # For non-zero error, average over two signs
            signs = [1, -1] if err > 0 else [0]
            f_alpha_vals = []
            f_loss_vals  = []

            for sign in signs:
                factor = 1 + sign * err

                # --- Case 1: error in alpha, loss reconstructed ---
                alpha_real = alpha_nom * factor
                # Reconstruct loss from data with real alpha_real and nominal T and loss
                loss_rec = reconstruct_loss(
                    r=R,
                    alpha_lo_theory=alpha_nom,   # assumed (nominal) alpha during reconstruction
                    alpha_lo_real=alpha_real,    # real alpha in experiment
                    t1=t_nom, t2=t_nom,          # real T (in this case equal to nominal, since we don't distort them)
                    t_know=t_nom,                 # known T (nominal)
                    loss1=loss_nom, loss2=loss_nom  # real loss (nominal, since we don't distort them in this scenario)
                )
                # Compute fidelity with reconstructed loss as known
                f_alpha = 1 - Fidelity_R_with_tree_povm(
                    ro_s_0=ro_s_0,
                    alpha_lo_theory=alpha_nom,
                    alpha_LO_real=alpha_real,
                    R=R,
                    t1=t_nom, t2=t_nom,
                    t_know=t_nom,
                    loss1_real=loss_nom, loss2_real=loss_nom,
                    loss_know=loss_rec
                )
                f_alpha_vals.append(f_alpha)

                # --- Case 2: error in loss, alpha reconstructed ---
                loss_real = [l * factor for l in loss_nom]
                # Reconstruct alpha from data with real loss_real and nominal T
                alpha_rec = reconstructed_alfa_pvm(
                    r=R,
                    alpha_lo_real=alpha_nom,      # real alpha (nominal, since we don't distort it)
                    t1=t_nom, t2=t_nom,
                    t_know=t_nom,
                    loss1=loss_real, loss2=loss_real,
                    loss_know=loss_nom             # during reconstruction we treat loss as known (nominal)
                )
                # Compute fidelity with reconstructed alpha as known
                f_loss = 1 - Fidelity_R_with_tree_povm(
                    ro_s_0=ro_s_0,
                    alpha_lo_theory=alpha_rec,    # use reconstructed alpha in the model
                    alpha_LO_real=alpha_nom,      # real alpha (nominal)
                    R=R,
                    t1=t_nom, t2=t_nom,
                    t_know=t_nom,
                    loss1_real=loss_real, loss2_real=loss_real,
                    loss_know=loss_nom
                )
                f_loss_vals.append(f_loss)

            # Average over two signs (if err=0, signs contains only 0, mean will be one value)
            fid_alpha[exp_count, i] = np.mean(f_alpha_vals)
            fid_loss[exp_count, i]  = np.mean(f_loss_vals)

            exp_count += 1
            current += 1

            if verbose:
                progress = (current / total_points) * 100
                print(f"\rOverall progress: {progress:.1f}%", end='', flush=True)

        if attempts == max_attempts:
            warnings.warn(f"Failed to gather the required number of experiments for error {err}. "
                          f"Used {exp_count} points.")

    if verbose:
        print()  # move to new line

    mean_alpha = np.mean(fid_alpha, axis=0)
    std_alpha  = np.std(fid_alpha, axis=0, ddof=1)
    mean_loss  = np.mean(fid_loss, axis=0)
    std_loss   = np.std(fid_loss, axis=0, ddof=1)

    results = {
        'err_vals': err_vals,
        'fidelity_alpha_mean': mean_alpha,
        'fidelity_alpha_std': std_alpha,
        'fidelity_loss_mean': mean_loss,
        'fidelity_loss_std': std_loss,
        'raw_data': {
            'fid_alpha': fid_alpha,
            'fid_loss': fid_loss
        }
    }

    if save_data:
        np.savez(f"{filename_base}.npz", **results)
        print(f"Data saved to {filename_base}.npz")

    _plot_two_cases(results, filename_base)
    return results


def _plot_two_cases(results: Dict[str, Any], filename_base: str = None) -> None:
    """Helper function to plot the graph."""
    err_percent = results['err_vals'] * 100
    plt.figure(figsize=(10, 6))

    plt.errorbar(err_percent, results['fidelity_alpha_mean'],
                 yerr=results['fidelity_alpha_std'],
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', markersize=6, linestyle='-', color='purple', linewidth=2,
                 label='Error in α LO (detector efficiencies are reconstructed)')

    plt.errorbar(err_percent, results['fidelity_loss_mean'],
                 yerr=results['fidelity_loss_std'],
                 capsize=5, capthick=2, elinewidth=2,
                 marker='s', markersize=6, linestyle='-', color='blue', linewidth=2,
                 label='Error in detector efficiency (α is reconstructed)')

    plt.xlabel('Relative parameter error (%)', fontsize=12)
    plt.ylabel('1 - Fidelity', fontsize=12)
    plt.title('Impact of calibration parameter errors on tomography accuracy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if filename_base:
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        print(f"Graph saved to {filename_base}.png")
    plt.show()


data2 = plot_fidelity_vs_mismatch_two_cases(start_err=0.0, stop_err=0.30, num_points=15, num_experiments=4,
                                            alpha_nom=2.0, loss_nom=0.55, t_nom=0.5, R=1,
                                            filename_base="test_two_cases", save_data=True
                                            )
print(data2)
_plot_two_cases(load_data=data2)