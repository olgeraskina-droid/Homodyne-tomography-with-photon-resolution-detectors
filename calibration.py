import numpy as np
from scipy.optimize import minimize, minimize_scalar
from config import dim_s
from utils import create_ro_s
from data_generation import create_data
from povm import calculate_q_with_loss, precompute_tensors

def reconstructed_alfa_pvm(r, alpha_lo_real, t1, t2, t_know, loss1, loss2, loss_know):
    """Reconstruct alpha LO from data using a known Fock state."""
    ro_s_0, _ = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(ro_s_0, alpha_lo_real, r, t1, t2, loss1, loss2)
    q_known = calculate_q_with_loss(t_know, loss_know)

    def neg_log_lik(alpha):
        tensors_w = precompute_tensors(alpha, r, q_known, q_known)
        logL = 0.0
        eps = 1e-12
        for theta_idx, k1, k2, cnt in data_pvm:
            W = tensors_w[theta_idx][:, :, k1, k2]
            prob = np.trace(ro_s_0 @ W).real
            prob = max(prob, eps)
            logL += cnt * np.log(prob)
        return -logL

    result = minimize_scalar(neg_log_lik, bounds=(0.1, 10), method='bounded')
    if result.success:
        return result.x
    else:
        print("Alpha optimization did not converge, returning 1.0")
        return 1.0

def reconstruct_loss(r, alpha_lo_theory, alpha_lo_real, t1, t2, t_know, loss1, loss2):
    """Reconstruct detector losses using a known Fock state and known alpha."""
    ro_s_0, _ = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(ro_s_0, alpha_lo_real, r, t1, t2, loss1, loss2)

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
    return result.x

def reconstruct_param_a_loss(r, alpha_lo_real, t1, t2, t_know, loss1, loss2):
    """Reconstruct alpha and loss simultaneously (not used in main flow)."""
    ro_s_0, _ = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(ro_s_0, alpha_lo_real, r, t1, t2, loss1, loss2)

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
    return result.x

def reconstruct_param(r, alpha_lo_theory, t1, t2, loss1, loss2):
    """Reconstruct alpha, T, and loss simultaneously (not used in main flow)."""
    ro_s_0, _ = create_ro_s(str_type='f', n=0, alpha_s=0.7)
    data_pvm = create_data(ro_s_0, alpha_lo_theory, r, t1, t2, loss1, loss2)

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
    return result.x