import numpy as np 
from mag_params import *
from branch_ranker import *
from llg_dynamics import *
from llg_plots import *
from num_exchange import *
from pathlib import Path
import sys

def print_candidate(row, K_scale_eV=None):
    print("=" * 70)

    print("r = kF R:", row["r"])
    print("y = qR/kF:", row["y"])
    print("kF allowed [1/angstrom]:", row["kF_allowed_min_inv_angstrom"], row["kF_allowed_max_inv_angstrom"])
    print("chosen kF [1/angstrom]:", row["kF_best_inv_angstrom"])
    print("R_best [nm]:", row["R_best_nm"])
    print("qR_best [1/angstrom]:", row["qR_best_inv_angstrom"])

    print("exchange_norm:", row["exchange_norm"])
    print("relative reduced diode metric:", row["relative_reduced_diode_metric"])
    print("full diode metric:", row["full_diode_metric"])
    print("eta:", row["eta1"], row["eta2"])
    print("root_sign:", row["root_sign"])
    print("suppress_direction:", row["suppress_direction"])
    print("B0:", row["B0"])
    print("omega:", row["omega"])
    print("period:", row["period"])
    print("steps_per_period:", row["steps_per_period"])
    print("recommended_n_steps:", row["recommended_n_steps"])
    print("ground:", row["is_ground_state"])
    print("stable:", row["is_locally_stable"])

    if K_scale_eV is not None:
        B0_eV = row["B0"]
        omega_eV = abs(row["omega"])
        print("B0 [eV]:", B0_eV)
        print("B0 [meV]:", 1e3 * B0_eV)
        print("B0 [T]:", eV_to_T(B0_eV))
        print("omega [eV]:", omega_eV)
        print("omega [meV]:", 1e3 * omega_eV)
        print("omega [GHz]:", eV_to_GHz(omega_eV))

def local_tensor_check(dyn):
    mag = dyn.mag_params
    R1 = dyn.R1
    R2 = dyn.R2
    eta1 = dyn.eta1
    eta2 = dyn.eta2

    K12_loc = R1 @ mag.K12 @ R2.T
    K21_loc = R2 @ mag.K21 @ R1.T
    A12_loc = R1 @ mag.A12 @ R2.T
    A21_loc = R2 @ mag.A21 @ R1.T

    c = mag.ex_and_damp_coeffs()
    J0 = c["J0"]
    J1 = c["J1"]
    D = c["D"]
    a0 = c["a0"]
    a1 = c["a1"]

    print("\nLOCAL TENSOR CHECK")
    print("=" * 70)
    print("eta1, eta2:", eta1, eta2)

    print("K12_loc:")
    print(K12_loc)
    print("K21_loc:")
    print(K21_loc)

    print("A12_loc:")
    print(A12_loc)
    print("A21_loc:")
    print(A21_loc)

    if eta1 == eta2:
        eta = eta1

        K_expected = np.array([
            [J0, -eta * D, 0.0],
            [eta * D, J0, 0.0],
            [0.0, 0.0, J1],
        ])

        A_expected = np.array([
            [a0, -eta * a1, 0.0],
            [eta * a1, a0, 0.0],
            [0.0, 0.0, mag.A12[1, 1]],
        ])

        print("max |K12_loc - K_expected|:", np.max(np.abs(K12_loc - K_expected)))
        print("max |A12_loc - A_expected|:", np.max(np.abs(A12_loc - A_expected)))

def main(
    mode="exact",
    material=None,
    S=1.5,
    theta=0.0,
    target_y=0.5,
    K_scale=5e-7,
    alpha=5e-3,
    dt=200,
    drive_amp=1e-10,
    handedness=+1,
    magnon_convention="psi",
    seed=1234,
    branch_type="parallel",
    candidate_index=0,
    run_sweeps=True,
    text_output=None,
    exchange_model="analytic",
    numerical_exchange_settings=None,
):
    from branch_ranker import DiodeRanker
    
    if text_output is not None:
        sys.stdout = open(str(text_output), "w")
    
    if numerical_exchange_settings is None:
        numerical_exchange_settings = ()

    print(f"Exchange model to be used: {exchange_model}")

    ranker = DiodeRanker(
        K_scale=K_scale,
        alpha_scale=alpha,
        S=S,
        theta=theta,
        x_min=0.1,
        x_max=50.0,
        n_x=5000,
    )

    if mode == "exact":
        rows, total_roots, diagnostics = ranker.rank_dimensionless_diode_candidates(
            #y_values=np.linspace(0.1, 0.75, 100),
            y_values=np.array([target_y], dtype=float),
            theta=theta,
            r_min=1.0,
            r_max=8.0,
            n_r=5000,
            kF_min_inv_angstrom=0.02,
            kF_max_inv_angstrom=0.30,
            kF_window_min_inv_angstrom=0.01,
            R_min_angstrom=3.0,
            R_max_angstrom=80.0,
            F_min=1e-5,
            exchange_min=None,
            B0_abs_min=None,
            B0_abs_max=5e-3,
            omega_abs_min=1e-8,
            omega_abs_max=5e-3,
            branch_type=branch_type,
            dt=None,
            n_cycles=40,
            min_steps_per_period=50,
            max_n_steps=2_000_000,
            return_diagnostics=True,
            exchange_model=exchange_model,
            numerical_exchange_settings=numerical_exchange_settings,
        )
        
        print("ranking mode: exact")
        print("total exact roots found:", total_roots)
        print("usable candidates:", len(rows))
        
        if len(rows) == 0:
            ranker.print_diagnostics(diagnostics)
            raise RuntimeError("No usable candidates found; see diagnostics above.")

    elif mode in ("material", "material_contrast"):
        if material is None:
            raise ValueError(
                "material must be provided when mode='material_contrast'"
            )

        rows = ranker.rank_material_contrast_candidates(
            materials=[material],
            theta=theta,
            R_min_angstrom=20.0,
            R_max_angstrom=100.0,
            n_R=1000,
            F_min=1e-5,
            exchange_min=None,
            B0_abs_min=None,
            B0_abs_max=5e-3,
            omega_abs_min=1e-8,
            omega_abs_max=5e-3,
            min_isolation=10.0,
            branch_type=branch_type,
            dt=dt,
            n_cycles=40,
            min_steps_per_period=50,
            max_n_steps=2_000_000,
            allowed_weight=0.25,
        )

        print("ranking mode: material_contrast")
        print("usable candidates:", len(rows))

    else:
        raise ValueError("mode must be 'exact' or 'material_contrast'")

    if not rows:
        raise RuntimeError("no usable ranked candidates found")

    row = rows[int(candidate_index)]
    print(list(row.keys()))

    print("=" * 70)
    print("selected candidate:", candidate_index)
    print("branch mode:", row.get("branch_mode"))
    print("y:", row["y"])
    print("r:", row["r"])
    print("R [nm]:", row.get("R_nm", row.get("R_best_nm")))
    print("eta1, eta2:", row["eta1"], row["eta2"])
    print("suppress direction:", row["suppress_direction"])
    print("B0 [eV]:", row["B0"])
    print("B0 [T]:", ranker.eV_to_T(row["B0"]))
    print("omega [eV]:", row["omega"])
    print("omega [GHz]:", ranker.eV_to_GHz(abs(row["omega"])))
    if "isolation_dB" in row:
        print("predicted isolation [dB]:", row["isolation_dB"])

    formulae = ranker.formulae(y=row["y"],
                               theta=row["theta"],
                               exchange_model=exchange_model,
                               r_min=0.01,
                               r_max=50.0,
                               n_r=5000,
                               settings=numerical_exchange_settings)
    mag = ranker.make_params(formulae, row["r"])
    xs_plot = np.linspace(1.0, 15.0, 5000)
    
    asymptotic_formulae = MagneticFormulae(
        y=row["y"],
        theta=row["theta"],
    )
    
    fig_exchange_comp, axes_exchange_comp = plot_exchange_comparison(
        numerical_formulae=formulae,
        asymptotic_formulae=asymptotic_formulae,
        xs=xs_plot,
        filename="../article/exchange_comparison.pdf",
    )    
    fig_coeffs, axes_coeffs = plot_candidate_exchange_and_damping(
        formulae=formulae,
        xs=xs_plot,
        filename="../article/candidate_exchange_and_damping.pdf",
    )
    diode_roots, _, _ = ranker.find_reduced_diode_roots(
       formulae=formulae,
       x_min=1.0,
       x_max=15.0,
       n_x=5000,
    )  
    fig_diode, ax_diode = plot_diode_condition_vs_x(
        formulae=formulae,
        xs=xs_plot,
        roots=diode_roots,
        x_candidate=row["r"],
        filename="../article/fig_of_merit.pdf",
    )

    mag.print_mag_params(mode="dimfull")
    mag.print_mag_params(mode="dimless")

    dyn = LLGDynamics(
        mag_params=mag,
        S_mag=S,
        alpha_loc=alpha,
        dt=row["dt"],
        T_eV=0.0,
        eta1=row["eta1"],
        eta2=row["eta2"],
        B0=row["B0"],
        omega=row["omega"],
        drive_amp=drive_amp,
        handedness=row["eta1"],
        drive_spin=0,
        magnon_convention=magnon_convention,
        ramp_cycles=20.0,
        ramp_type="cosine",
        seed=seed,
    )

    mode_dir = "material" if mode in ("material", "material_contrast") else "exact"
    plot_dir = Path("llg_plots") / mode_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    result = dyn.run_diode_pair(
        n_cycles=600,
        discard_fraction=0.875,
        store_fields=False,
        verbose=False,
    )

    
    out_21 = result["outputs"][0]
    out_12 = result["outputs"][1]
    


    timeseries_data = timeseries_data_from_result(
        result=result,
        suppress_direction=row["suppress_direction"],
    )


    fig_ts, axes_ts = plot_diode_timeseries(
        timeseries_data,
        filename="../article/llg_diode_timeseries.pdf",
    )

    omega_min = 0.8 * dyn.omega
    omega_max = 1.2 * dyn.omega
    
    if omega_min > omega_max:
        omega_min, omega_max = omega_max, omega_min
    
    chi_analytic = dyn.analytic_susceptibility(
        omega_min=omega_min,
        omega_max=omega_max,
        n_omega=2000,
    )
 
    fig_chi, ax_chi = plot_analytic_susceptibility(
        chi_analytic,
        filename="../article/susceptibility.pdf",
        x_axis="omega_drive",
        x_min=0.0,
        x_max=3.0,
    )
    omega_values = dyn.omega * np.linspace(
        0.8,
        1.2,
        121,
    )
    
    print("computing numerical swept susceptibility.")
    chi_llg_sweep = dyn.susceptibility_drive_sweep(
        omega_values=omega_values,
        n_cycles=300,
        discard_fraction=0.8,
        drive_amp=drive_amp,
    )
    
    fig_llg_sweep, ax_llg_sweep = plot_llg_swept_susceptibility(
        chi_llg_sweep,
        omega_ref=dyn.omega,
        filename="../article/numerical_susceptibility.pdf",
    )

    outputs = {
        "ranker": ranker,
        "rows": rows,
        "row": row,
        "formulae": formulae,
        "mag": mag,
        "dyn": dyn,
        "result": result,
        "timeseries_data": timeseries_data,
        "plot_dir": plot_dir,
        "fig_timeseries": fig_ts,
    }

    if run_sweeps:
        print("Testing amplitude sweep")
        drive_amplitude_sweep = dyn.test_drive_amplitude_sweep(
            amplitudes=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11),
            n_cycles=120,
            discard_fraction=0.75,
            print_results=True,
        )

        for sweep_row in drive_amplitude_sweep:
            sweep_row["drive_amp_over_B0"] = (
                abs(sweep_row["drive_amp"]) / max(abs(sweep_row["B0"]), 1e-300)
            )
    
        fig_amp, ax_amp = plot_response_sweep(
            drive_amplitude_sweep,
            x_key="drive_amp_over_B0",
            xlabel=r"$B_1 / |B_0|$",
            logx=True,
            filename="../article/llg_drive_amplitude_sweep.pdf",
        )
        
        #print("Testing timestep sweep")
        #timestep_sweep = dyn.test_timestep_sweep(
        #    dts=(1000.0, 500.0, 250.0, 100.0),
        #    n_cycles=120,
        #    drive_amp=drive_amp,
        #    discard_fraction=0.75,
        #    print_results=True,
        #)

        #fig_dt, ax_dt = plot_response_sweep(
        #    timestep_sweep,
        #    x_key="dt",
        #    xlabel=r"$\Delta t$",
        #    logx=True,
        #    filename=plot_dir / "llg_timestep_sweep.png",
        #)
        #
        #print("Testing transient sweep")
        #transient_sweep = dyn.test_transient_sweep(
        #    n_cycles_values=(40, 80, 120, 200, 400),
        #    dt=dt,
        #    drive_amp=drive_amp,
        #    discard_fraction=0.75,
        #    print_results=True,
        #)

        #fig_tr, ax_tr = plot_response_sweep(
        #    transient_sweep,
        #    x_key="n_cycles",
        #    xlabel="number of drive cycles",
        #    logx=False,
        #    filename=plot_dir / "llg_transient_sweep.png",
        #)
        #
        #print("Testing discard fraction")
        #discard_fraction_sweep = dyn.test_discard_fraction_sweep(
        #    discard_fractions=(0.25, 0.5, 0.75, 0.875),
        #    n_cycles=120,
        #    dt=dt,
        #    drive_amp=drive_amp,
        #    print_results=True,
        #)

        #fig_df, ax_df = plot_response_sweep(
        #    discard_fraction_sweep,
        #    x_key="discard_fraction",
        #    xlabel="discard fraction",
        #    logx=False,
        #    filename=plot_dir / "llg_discard_fraction_sweep.png",
        #)

        outputs.update({
            "drive_amplitude_sweep": drive_amplitude_sweep,
            "fig_drive_amplitude_sweep": fig_amp,
           # "timestep_sweep": timestep_sweep,
           # "fig_timestep_sweep": fig_dt,
           # "transient_sweep": transient_sweep,
           # "fig_transient_sweep": fig_tr,
           # "discard_fraction_sweep": discard_fraction_sweep,
           # "fig_discard_fraction_sweep": fig_df,
        })

    print("=" * 70)
    print("plots written to:", plot_dir.resolve())

    return outputs

if __name__ == "__main__":
    outputs = main(
        mode="exact",
        run_sweeps=True,
        handedness=+1,
        exchange_model="numerical"
        #text_output="output_diode_1.txt"
    )
    #outputs = main(
    #    mode="exact",
    #    run_sweeps=False,
    #    handedness=+1,
    #    exchange_model="analytic"
    #    #text_output="output_diode_1.txt"
    #)
    # For material-constrained runs, use this instead:
    #
    # outputs = main(
    #     mode="material_contrast",
    #     material={
    #         "name": "BiAg2-like",
    #         "kF_inv_angstrom": 0.23,
    #         "qR_inv_angstrom": 0.11,
    #     },
    #     run_sweeps=True,
    # )
