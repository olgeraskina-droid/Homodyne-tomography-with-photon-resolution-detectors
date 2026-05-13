import numpy as np
from scipy.optimize import minimize, basinhopping, OptimizeResult
from qutip import Qobj, fidelity as qutip_fidelity
from config import dim_s
from data_generation import create_data
from povm import precompute_tensors, calculate_q_with_loss

def params_to_rho(p, dim):
    """Convert parameter vector to density matrix via Cholesky-like decomposition."""
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

def rho_to_params(rho, dim):
    """Convert density matrix to parameter vector (Cholesky if possible)."""
    rho_reg = rho + 1e-12 * np.eye(dim)
    try:
        L = np.linalg.cholesky(rho_reg)
    except np.linalg.LinAlgError:
        from scipy.linalg import sqrtm
        L = sqrtm(rho_reg)
        # Fallback: return max mixed state parameters
        p0 = np.zeros(dim**2)
        p0[:dim] = np.log(np.ones(dim) / dim)
        p0[dim:] = np.random.normal(scale=0.1, size=dim*(dim-1))
        return p0
    trace = np.trace(L @ L.conj().T).real
    L /= np.sqrt(trace)
    p = []
    idx = 0
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                p.append(L[i, i].real)
                idx += 1
            else:
                p.append(L[i, j].real)
                p.append(L[i, j].imag)
                idx += 2
    return np.array(p)

def neg_log_likelihood_pvm(p, data, tensors_w, *args):
    """Negative log-likelihood for PVM measurements."""
    rho = params_to_rho(p, dim_s)
    logL = 0.0
    eps = 1e-12
    for theta_idx, k1, k2, cnt in data:
        W = tensors_w[theta_idx][:, :, k1, k2]
        prob = np.trace(rho @ W).real
        prob = max(prob, eps)
        logL += cnt * np.log(prob)
    return -logL

def rrhor_pvm(data, dim_s, tensors_w, max_iter=150, tol=1e-12):
    """RrhoR iterative MLE algorithm."""
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

def adam_minimize(loss_fn, p0, args, max_iter=500, lr=0.01):
    """Adam optimizer for MLE."""
    p = p0.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    best_p = p.copy()
    best_loss = np.inf

    data_full = args[0]  # X_theta

    for t in range(1, max_iter+1):
        # Mini-batch: 50% of data
        batch_size = max(1, len(data_full) // 2)
        idx = np.random.choice(len(data_full), batch_size, replace=False)
        data_batch = [data_full[i] for i in idx]

        grad = np.zeros_like(p)
        eps_grad = 1e-8
        loss0 = loss_fn(p, data_batch, *args[1:])

        for i in range(len(p)):
            p_step = p.copy()
            p_step[i] += eps_grad
            loss1 = loss_fn(p_step, data_batch, *args[1:])
            grad[i] = (loss1 - loss0) / eps_grad

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)

        if t % 50 == 0:
            loss_full = loss_fn(p, data_full, *args[1:])
            if loss_full < best_loss:
                best_loss = loss_full
                best_p = p.copy()

    return OptimizeResult(x=best_p, fun=best_loss, success=True)

def Fidelity_R_with_tree_povm(R, ro_s_0, alpha_lo_theory, alpha_LO_real,
                              t1, t2, t_know, loss1_real, loss2_real, loss_know):
    """Compute fidelity between true and reconstructed state using tree POVM."""
    data_pvm = create_data(ro_s_0, alpha_LO_real, R, t1, t2, loss1_real, loss2_real)
    q1_known = calculate_q_with_loss(t_know, loss_know)
    q2_known = q1_known
    tensors_w = precompute_tensors(alpha_lo_theory, R, q1_known, q2_known)

    p0_RroR = rho_to_params(rrhor_pvm(data_pvm, dim_s, tensors_w, max_iter=100), dim_s)
    print('init. rho')
    print(np.round(ro_s_0, 4))

    f_best = 0
    fun_best = np.inf
    count_start = 0
    while count_start < 1:   # original only 0 executed, kept for structure
        if count_start == 0:
            result = minimize(neg_log_likelihood_pvm, p0_RroR,
                              args=(data_pvm, tensors_w),
                              method='L-BFGS-B')
        else:
            # (other strategies in original, not needed for typical run)
            break
        rec_rho = params_to_rho(result.x, dim_s)
        f = float(qutip_fidelity(Qobj(ro_s_0), Qobj(rec_rho / np.trace(rec_rho))))
        print(' ')
        print('rec rho')
        print(np.round(rec_rho, 4))
        if f > f_best:
            f_best = f
        if result.fun < fun_best:
            best_res_fun = result.fun
            f_best_fun = f
        count_start += 1

    print('f_best_fun = ' + str(f_best_fun))
    if np.abs(f_best - f_best_fun) > 0.015:
        print('DID NOT MATCH!!!!!!!!!!!!!!!! best fun fidelity: ' + str(f_best_fun))
    return f_best