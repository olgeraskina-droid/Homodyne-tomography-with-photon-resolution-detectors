import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
from config import (alpha_LO_theory, loss_KNOW, T_know, R,
                    count_theta, iterations, d, dim_s)
from utils import create_ro_s
from tomography import Fidelity_R_with_tree_povm
from calibration import reconstructed_alfa_pvm, reconstruct_loss

def create_pic_f_t(start_T_sr, stop_T_sr, count_T, count_exp,
                   alpha_lo_theory=alpha_LO_theory, loss_know=loss_KNOW,
                   t_know=T_know, r=R):
    if t_know is None:
        t_know = T_know
    fidelity_err = np.zeros((count_exp, count_T), dtype=float)

    target_means = np.linspace(start_T_sr, stop_T_sr, count_T)
    T1_matrix = [[round(mean, 3) for _ in range(7)] for mean in target_means]

    for i in range(count_exp):
        for j in range(count_T):
            ro_s_0, _ = create_ro_s('hk', n=0, alpha_s=0)
            if ro_s_0[2][2] > 0.17: continue
            fidelity_err[i][j] = 1 - Fidelity_R_with_tree_povm(
                ro_s_0=ro_s_0, alpha_lo_theory=alpha_lo_theory,
                alpha_LO_real=alpha_lo_theory, R=r,
                t1=T1_matrix[j], t2=T1_matrix[j], t_know=T_know,
                loss1_real=loss_KNOW, loss2_real=loss_KNOW, loss_know=loss_know
            )
            progress = ((j + 1) / (count_T * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)
    T_otn_err = [100 * abs(1 - sum(T)/len(T)/t_know[0]) for T in T1_matrix]

    plt.figure(figsize=(10,6))
    plt.errorbar(T_otn_err, means, yerr=std_errors,
                 capsize=5, capthick=2, elinewidth=2,
                 marker='o', linestyle='-', linewidth=2)
    plt.xlabel('T error (%)')
    plt.ylabel('1 - Fidelity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "f_t.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPic save in: {filepath}")
    plt.close()

def create_pic_f_detect_loss(start_loss, stop_loss, count_loss, count_exp,
                             alpha_lo_theory=alpha_LO_theory, loss_know=loss_KNOW,
                             t_know=T_know, r=R):
    fidelity_err = np.zeros((count_exp, count_loss), dtype=float)

    step = (stop_loss - start_loss) / count_loss
    arrays_data = []
    for i in range(count_loss):
        segment_start = start_loss + i * step
        segment_end = start_loss + (i + 1) * step
        loss = [round(random.uniform(segment_start, segment_end), 3) for _ in range(8)]
        mean_value = sum(loss) / len(loss)
        arrays_data.append((loss, mean_value))
    arrays_data.sort(key=lambda x: x[1])
    loss1_matrix = [loss for loss, _ in arrays_data]

    for i in range(count_exp):
        for j in range(count_loss):
            ro_s_0, _ = create_ro_s('hk', n=0, alpha_s=0)
            if ro_s_0[2][2] > 0.17: continue
            fidelity_err[i][j] = 1 - Fidelity_R_with_tree_povm(
                ro_s_0=ro_s_0, alpha_lo_theory=alpha_lo_theory,
                alpha_LO_real=alpha_lo_theory, R=r,
                t1=T_know, t2=T_know, t_know=T_know,
                loss1_real=loss1_matrix[j], loss2_real=loss1_matrix[j],
                loss_know=loss_know
            )
            progress = ((j + 1) / (count_loss * count_exp) + i / count_exp) * 100
            print(f"\r-_- {round(progress)}%", end='', flush=True)

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)
    loss_otn_err = [100 * abs(1 - sum(l)/len(l)/loss_know[0]) for l in loss1_matrix]

    plt.figure(figsize=(10,6))
    plt.errorbar(loss_otn_err, means*100, yerr=std_errors, capsize=5, marker='o')
    plt.xlabel('loss error (%)')
    plt.ylabel('1 - Fidelity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"loss_alpha_known_loss_{loss_know[1]}.png"
    plt.savefig(os.path.join(script_dir, filename), dpi=300)
    print(f"\nGraph saved to: {os.path.join(script_dir, filename)}")
    plt.close()

# (Other plotting functions follow the same pattern - I'll provide the most important ones for brevity,
# but the complete set is in the original. Below are the two key combined plots.)
# ... [ create_pic_f_alpha, create_pic_f_error, create_pic_reconstruct_alpha_t_loss, etc. ] ...

def pic_f_a_with_reconstruction_loss(start_a, stop_a, count_a, count_exp,
                                     alpha_lo_theory=alpha_LO_theory, R=R):
    loss_true = [round(random.uniform(0.5, 0.65), 3) for _ in range(8)]
    t_true = [round(random.uniform(0.42, 0.58), 3) for _ in range(7)]
    fidelity_err = np.zeros((count_exp, count_a), dtype=float)
    alpha_arr = np.linspace(start_a, stop_a, count_a)

    for j, alpha_real in enumerate(alpha_arr):
        loss_rec = reconstruct_loss(r=R, alpha_lo_theory=alpha_LO_theory,
                                    alpha_lo_real=alpha_real, t1=t_true, t2=t_true,
                                    t_know=T_know, loss1=loss_true, loss2=loss_true)
        for i in range(count_exp):
            ro_s_0, _ = create_ro_s('hk', n=0, alpha_s=0)
            if ro_s_0[2][2] > 0.1 or ro_s_0[0][0] < 0.5:
                continue
            fidelity_err[i][j] = 1 - Fidelity_R_with_tree_povm(
                ro_s_0=ro_s_0, alpha_lo_theory=alpha_LO_theory,
                alpha_LO_real=alpha_real, R=R, t1=t_true, t2=t_true,
                t_know=T_know, loss1_real=loss_true, loss2_real=loss_true,
                loss_know=loss_rec
            )
            progress = ((i+1)/(count_exp*count_a) + j/count_a)*100
            print(f"\r-_- {round(progress)}%", end='', flush=True)

    means = np.mean(fidelity_err, axis=0)
    std_errors = np.std(fidelity_err, axis=0, ddof=1)

    x_over, means_over, std_over = [], [], []
    x_under, means_under, std_under = [], [], []
    for j, alpha in enumerate(alpha_arr):
        x_val = 100 * abs(1 - alpha/alpha_lo_theory)
        if alpha >= alpha_lo_theory:
            x_over.append(x_val); means_over.append(means[j]); std_over.append(std_errors[j])
        if alpha <= alpha_lo_theory:
            x_under.append(x_val); means_under.append(means[j]); std_under.append(std_errors[j])

    plt.figure(figsize=(12,7))
    plt.errorbar(x_over, means_over, yerr=std_over, marker='o', color='red', label='Overestimation (α > α₀)')
    plt.errorbar(x_under, means_under, yerr=std_under, marker='s', color='blue', label='Underestimation (α < α₀)')
    plt.xlabel('Relative error of α LO |1 − α/α₀| (%)')
    plt.ylabel('1 − Fidelity')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'f_alpha_withrecloss.png'), dpi=300)
    print(f"Graph saved to {os.path.join(script_dir, 'f_alpha_withrecloss.png')}")
    plt.show()

def _plot_two_cases(results, filename_base=None):
    err_percent = results['err_vals'] * 100
    plt.figure(figsize=(10,6))
    plt.errorbar(err_percent, results['fidelity_alpha_mean'], yerr=results['fidelity_alpha_std'],
                 capsize=5, marker='o', color='purple',
                 label='Error in α LO (detector efficiencies reconstructed)')
    plt.errorbar(err_percent, results['fidelity_loss_mean'], yerr=results['fidelity_loss_std'],
                 capsize=5, marker='s', color='blue',
                 label='Error in detector efficiency (α reconstructed)')
    plt.xlabel('Relative parameter error (%)')
    plt.ylabel('1 − Fidelity')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if filename_base:
        plt.savefig(f"{filename_base}.png", dpi=300)
        print(f"Graph saved to {filename_base}.png")
    plt.show()

def plot_fidelity_vs_mismatch_two_cases(
    start_err=0.0, stop_err=0.2, num_points=10, num_experiments=5,
    alpha_nom=None, loss_nom=None, t_nom=None, R=None,
    filename_base="fidelity_two_cases", save_data=True,
    load_data=None, verbose=True, **kwargs
):
    if load_data is not None:
        _plot_two_cases(load_data)
        return load_data

    global alpha_LO_theory, loss_KNOW, T_know
    if alpha_nom is None: alpha_nom = alpha_LO_theory
    if loss_nom is None: loss_nom = loss_KNOW
    if t_nom is None: t_nom = T_know
    if R is None: R = 1
    if isinstance(loss_nom, (int, float)): loss_nom = [float(loss_nom)]*8
    if isinstance(t_nom, (int, float)): t_nom = [float(t_nom)]*7

    err_vals = np.linspace(start_err, stop_err, num_points)
    fid_alpha = np.zeros((num_experiments, num_points))
    fid_loss  = np.zeros((num_experiments, num_points))

    for i, err in enumerate(err_vals):
        if verbose: print(f"\nPoint {i+1}/{num_points}, error = {err*100:.2f}%")
        exp_count = 0
        while exp_count < num_experiments:
            ro_s_0, _ = create_ro_s('hk', n=0, alpha_s=0)
            if ro_s_0[2][2] > 0.1 or ro_s_0[0][0] < 0.5: continue
            signs = [1, -1] if err > 0 else [0]
            f_alpha_vals, f_loss_vals = [], []
            for sign in signs:
                factor = 1 + sign * err
                # Case 1: alpha mismatch, loss reconstructed
                alpha_real = alpha_nom * factor
                loss_rec = reconstruct_loss(r=R, alpha_lo_theory=alpha_nom, alpha_lo_real=alpha_real,
                                            t1=t_nom, t2=t_nom, t_know=t_nom,
                                            loss1=loss_nom, loss2=loss_nom)
                f_alpha = 1 - Fidelity_R_with_tree_povm(
                    ro_s_0=ro_s_0, alpha_lo_theory=alpha_nom, alpha_LO_real=alpha_real,
                    R=R, t1=t_nom, t2=t_nom, t_know=t_nom,
                    loss1_real=loss_nom, loss2_real=loss_nom, loss_know=loss_rec)
                f_alpha_vals.append(f_alpha)

                # Case 2: loss mismatch, alpha reconstructed
                loss_real = [l * factor for l in loss_nom]
                alpha_rec = reconstructed_alfa_pvm(r=R, alpha_lo_real=alpha_nom,
                                                   t1=t_nom, t2=t_nom, t_know=t_nom,
                                                   loss1=loss_real, loss2=loss_real, loss_know=loss_nom)
                f_loss = 1 - Fidelity_R_with_tree_povm(
                    ro_s_0=ro_s_0, alpha_lo_theory=alpha_rec, alpha_LO_real=alpha_nom,
                    R=R, t1=t_nom, t2=t_nom, t_know=t_nom,
                    loss1_real=loss_real, loss2_real=loss_real, loss_know=loss_nom)
                f_loss_vals.append(f_loss)

            fid_alpha[exp_count, i] = np.mean(f_alpha_vals)
            fid_loss[exp_count, i]  = np.mean(f_loss_vals)
            exp_count += 1

    results = {
        'err_vals': err_vals,
        'fidelity_alpha_mean': np.mean(fid_alpha, axis=0),
        'fidelity_alpha_std': np.std(fid_alpha, axis=0, ddof=1),
        'fidelity_loss_mean': np.mean(fid_loss, axis=0),
        'fidelity_loss_std': np.std(fid_loss, axis=0, ddof=1),
        'raw_data': {'fid_alpha': fid_alpha, 'fid_loss': fid_loss}
    }
    if save_data:
        np.savez(f"{filename_base}.npz", **results)
        print(f"Data saved to {filename_base}.npz")
    _plot_two_cases(results, filename_base)
    return results