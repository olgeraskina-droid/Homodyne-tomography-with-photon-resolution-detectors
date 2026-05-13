from plotting import plot_fidelity_vs_mismatch_two_cases

if __name__ == "__main__":
    data2 = plot_fidelity_vs_mismatch_two_cases(
        start_err=0.0, stop_err=0.30, num_points=15, num_experiments=4,
        alpha_nom=2.0, loss_nom=0.55, t_nom=0.5, R=1,
        filename_base="test_two_cases", save_data=True
    )
    print(data2)