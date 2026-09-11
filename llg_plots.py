import numpy as np 
import matplotlib.pyplot as plt

def paper_plot_style(label_size=16, tick_size=13, legend_size=12):
    plt.rcParams.update({
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
    })


def exchange_coeffs(formulae, xs):
    return {
        "J0": formulae.J_xx(xs),
        "J1": formulae.J_yy(xs),
        "D": formulae.J_xz(xs),
    }


def damping_coeffs(formulae, xs):
    return {
        "a0": formulae.A_xx(xs) / 4.0,
        "a1": formulae.A_xz(xs) / 4.0,
        "a2": formulae.A_yy(xs) / 4.0,
    }

def results_to_arrays(results, x_key):
    x = np.asarray([row[x_key] for row in results], dtype=float)

    A12 = np.asarray([row["A12"] for row in results], dtype=float)
    A21 = np.asarray([row["A21"] for row in results], dtype=float)

    ratio_21_12 = np.asarray([row["A21_over_A12"] for row in results], dtype=float)
    ratio_12_21 = np.asarray([row["A12_over_A21"] for row in results], dtype=float)

    return x, A12, A21, ratio_21_12, ratio_12_21


def plot_response_sweep(results, x_key, xlabel, title=None, logx=False, filename=None):
    x, A12, A21, ratio_21_12, ratio_12_21 = results_to_arrays(results, x_key)

    fig, ax_ratio = plt.subplots()

    ax_ratio.scatter(x, ratio_21_12, marker="o", label=r"$A_{21}/A_{12}$", c='red')
    ax_ratio.scatter(x, ratio_12_21, marker="s", label=r"$A_{12}/A_{21}$", c='blue')
    ax_ratio.axhline(1.0, linestyle=":", linewidth=1.0, c='black')

    ax_ratio.set_ylabel(r"Ratio")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_yscale("log")
    ax_ratio.legend(frameon=False)

    if logx:
        ax_ratio.set_xscale("log")

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax_ratio


def timeseries_data_from_result(result, suppress_direction):
    out_21 = result["outputs"][0]
    out_12 = result["outputs"][1]

    t0 = int(result["discard_fraction"] * len(out_21["t"]))
    t = out_21["t"][t0:] / result["period"]

    response_21 = np.real(out_21["psi2"][t0:])
    response_12 = np.real(out_12["psi1"][t0:])

    if suppress_direction == "12":
        allowed = response_12
        suppressed = response_21
        allowed_label = r"$\chi_{12}$"
        suppressed_label = r"$\chi_{21}$"
        driven_spin = 1
        respond_spin = 2
    elif suppress_direction == "21":
        allowed = response_21
        suppressed = response_12
        allowed_label = r"$\chi_{21}$"
        suppressed_label = r"$\chi_{12}$"
        driven_spin = 2 
        respond_spin = 1
    else:
        raise ValueError("suppress_direction must be '12' or '21'")

    return {
        "t": t,
        "response_21": response_21,
        "response_12": response_12,
        "allowed": allowed,
        "suppressed": suppressed,
        "allowed_label": allowed_label,
        "suppressed_label": suppressed_label,
        "driven_spin": driven_spin,
        "respond_spin": respond_spin,
        "A21": result["A21"],
        "A12": result["A12"],
        "A21_over_A12": result["A21_over_A12"],
        "A12_over_A21": result["A12_over_A21"],
        "suppress_direction": suppress_direction,
        "discard_fraction": result["discard_fraction"],
        "n_cycles": result["n_cycles"],
        "drive_amp": result["drive_amp"],
    }


def plot_diode_timeseries(timeseries_data, filename=None):
    t = timeseries_data["t"]
    B1_amp = timeseries_data["drive_amp"]
    allowed = timeseries_data["allowed"] / B1_amp
    suppressed = timeseries_data["suppressed"] / B1_amp

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(7.2, 6.0),
        sharex=True,
    )
    driven_spin = timeseries_data["driven_spin"]
    respond_spin = timeseries_data["respond_spin"]

    ax_allowed, ax_suppressed_same, ax_suppressed_zoom = axes

    ax_allowed.plot(t, allowed, c='red')
    ax_allowed.set_ylabel(rf"$\mathrm{{Re}}(\psi_{{{driven_spin}}}) / B_1$")
    #ax_allowed.set_title(f"Allowed channel {timeseries_data['allowed_label']}")

    ax_suppressed_same.plot(t, suppressed, c='blue')
    ax_suppressed_same.set_ylabel(rf"$\mathrm{{Re}}(\psi_{{{respond_spin}}}) / B_1$")
    #ax_suppressed_same.set_title(
    #    f"Suppressed channel {timeseries_data['suppressed_label']}, same scale"
    #)
    ax_suppressed_same.set_ylim(ax_allowed.get_ylim())

    ax_suppressed_zoom.plot(t, suppressed, c='blue')
    ax_suppressed_zoom.set_ylabel(rf"$\mathrm{{Re}}(\psi_{{{respond_spin}}}) / B_1$")
    #ax_suppressed_zoom.set_title(
    #    f"Suppressed channel {timeseries_data['suppressed_label']}, zoomed"
    #)
    ax_suppressed_zoom.set_xlabel(r"$t/T_{\mathrm{drive}}$")

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, axes


def fft_data_from_result(dyn, result):
    out_21 = result["outputs"][0]
    out_12 = result["outputs"][1]

    chi_21 = dyn.copy_with(drive_spin=0).susceptibility_fft(
        timeseries=out_21,
        discard_fraction=result["discard_fraction"],
        response_spin=1,
    )

    chi_12 = dyn.copy_with(drive_spin=1).susceptibility_fft(
        timeseries=out_12,
        discard_fraction=result["discard_fraction"],
        response_spin=0,
    )

    return {
        "chi_21": chi_21,
        "chi_12": chi_12,
    }




def plot_fft_susceptibility(fft_data, filename=None, x_min=0.05):
    chi_12 = fft_data["chi_12"]
    chi_21 = fft_data["chi_21"]

    fig, ax_chi = plt.subplots(figsize=(7.0, 4.8))

    x12 = chi_12["frequency_cycles"]
    x21 = chi_21["frequency_cycles"]

    y12 = chi_12["abs_normalised_response_fft"]
    y21 = chi_21["abs_normalised_response_fft"]

    m12 = x12 >= x_min
    m21 = x21 >= x_min

    ax_chi.plot(x12[m12], y12[m12], label=r"$|\psi_1(\omega)|/|b_2(\omega_d)|$", c='red')
    ax_chi.plot(x21[m21], y21[m21], label=r"$|\psi_2(\omega)|/|b_1(\omega_d)|$", c='blue')
    ax_chi.axvline(1.0, linestyle="--", linewidth=1.0, c='black')

    ax_chi.set_xlim(x_min, 3.0)
    ax_chi.set_ylabel("normalised response FFT")
    ax_chi.set_xlabel(r"$\omega/\omega_{\mathrm{drive}}$")
    ax_chi.set_yscale("log")
    ax_chi.legend(frameon=False)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax_chi

def plot_analytic_susceptibility(chi_data, filename=None, x_axis="omega_drive", x_min=None, x_max=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    if x_axis == "omega":
        x = chi_data["omega"]
        xlabel = r"$\omega$"
    elif x_axis == "omega_drive":
        x = chi_data["num_cycles"]
        xlabel = r"$\omega/\omega_{\mathrm{drive}}$"
    else:
        raise ValueError("x_axis must be 'omega' or 'omega_drive'")

    ax.plot(x, chi_data["abs_chi12"], label=r"$|\chi_{12}(\omega)|$", c="red")
    ax.plot(x, chi_data["abs_chi21"], label=r"$|\chi_{21}(\omega)|$", c="blue")

    if x_axis == "omega_drive":
        ax.axvline(1.0, linestyle="--", linewidth=1.0, c="black")

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|\chi(\omega)|$")
    ax.set_xlim([0.8, 1.2])
    ax.set_yscale("log")
    ax.legend(frameon=False)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax

def plot_candidate_exchange_and_damping(formulae, xs, x_candidate=None, filename=None):
    paper_plot_style()

    xs = np.asarray(xs, dtype=float)
    exch = exchange_coeffs(formulae, xs)
    damp = damping_coeffs(formulae, xs)

    colors = {
        "J0": "red",
        "J1": "blue",
        "D": "gold",
        "a0": "red",
        "a2": "blue",
        "a1": "gold",
    }

    fig, axes = plt.subplots(nrows=2,  sharex=True)
    ax_ex, ax_damp = axes

    ax_ex.plot(xs, exch["J0"], c=colors["J0"], label=r"$J_0$")
    ax_ex.plot(xs, exch["J1"], c=colors["J1"], label=r"$J_1$")
    ax_ex.plot(xs, exch["D"], c=colors["D"], label=r"$D$")
    ax_ex.set_xlabel(r"$k_F R$")
    ax_ex.set_ylabel(f"Exchange $ / K$")
    ax_ex.legend(frameon=False)

    ax_damp.plot(xs, damp["a0"], c=colors["a0"], label=r"$\alpha_0$")
    ax_damp.plot(xs, damp["a1"], c=colors["a1"], label=r"$\alpha_1$")
    ax_damp.plot(xs, damp["a2"], c=colors["a2"], label=r"$\alpha_2$")
    ax_damp.set_xlabel(r"$k_F R$")
    ax_damp.set_ylabel(r"Damping $/ \alpha$")
    ax_damp.legend(frameon=False)

    if x_candidate is not None:
        for ax in axes:
            ax.axvline(x_candidate, c="black", ls="--", lw=1.0)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, axes

def plot_diode_condition_vs_x(formulae, xs, roots=None, x_candidate=None, filename=None):
    paper_plot_style()

    xs = np.asarray(xs, dtype=float)
    vals = np.asarray([formulae.diode_condition(float(x)) for x in xs])

    fig, ax = plt.subplots()#figsize=(7.0, 4.6))
    roots_in_view = roots[
        (roots >= np.min(xs))
        & (roots <= np.max(xs))
    ]
    
    ax_ins = ax.inset_axes([0.5, 0.1, 0.3, 0.3])

    ax.plot(xs, vals, c="red")
    ax.axhline(0.0, c="black", ls="--", lw=1.0)
    ax.scatter(roots_in_view, np.zeros_like(roots_in_view), s=35, c='green', marker='x', zorder=5)

    ax.scatter([x_candidate], [0.0], s=70, c='gold', edgecolors='black', zorder=6)
    
    ax_ins.plot(xs, vals, c="red")
    ax_ins.scatter(roots_in_view, np.zeros_like(roots_in_view), s=35, c='green', marker='x')
    ax_ins.scatter([x_candidate], [0.0], s=70, c='gold', edgecolors='black', zorder=6)
    ax_ins.axhline(0.0, c='black', ls='--', lw=1.0)

    if x_candidate is not None:
        ax.axvline(x_candidate, c="black", ls=":", lw=1.0)
    
    ax.set_xlabel(r"$k_F R$")
    ax.set_ylabel(r"$D\alpha_1 + J_0\alpha_0$")
    #ax.set_xlim([np.min(xs), 5.0])
    ax.set_ylim([-0.25, 0.25])

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax
def plot_exchange_comparison(numerical_formulae, asymptotic_formulae, xs, filename=None):
    paper_plot_style(label_size=11, tick_size=9, legend_size=9)

    xs = np.asarray(xs, dtype=float)

    components = [
        (r"$J_0 / K$", "J_xx"),
        (r"$J_1 / K$", "J_yy"),
        (r"$D / K$", "J_xz"),
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(7.0, 2.6),
        sharex=True,
    )

    handles = None
    labels = None

    for ax, (ylabel, method_name) in zip(axes, components):
        f_num = getattr(numerical_formulae, method_name)
        f_asy = getattr(asymptotic_formulae, method_name)

        y_num = f_num(xs)
        y_asy = f_asy(xs)
        y_weak = f_asy(xs, weak_soc=True)

        ax.plot(xs, y_num, c="red", label="numerical", lw=1.25)
        ax.plot(xs, y_asy, c="blue", label="asymptotic", lw=1.25)
        ax.plot(xs, y_weak, c="gold", label="weak-SOC asymptotic", lw=1.25)

        ax.set_xlabel(r"$k_F R$")
        ax.set_ylabel(ylabel)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.8,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82], pad=0.25, w_pad=0.7)

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.02)

    return fig, axes

def plot_llg_swept_susceptibility(chi_data, omega_ref, filename=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    x = chi_data["omega"] / omega_ref

    ax.plot(x, chi_data["abs_chi12"], c="red", label=r"$|\chi_{12}^{\mathrm{LLG}}|$")
    ax.plot(x, chi_data["abs_chi21"], c="blue", label=r"$|\chi_{21}^{\mathrm{LLG}}|$")

    ax.axvline(1.0, linestyle="--", linewidth=1.0, c="black")
    ax.set_xlabel(r"$\omega/\omega_{\mathrm{drive}}$")
    ax.set_ylabel(r"$|\chi(\omega)|$")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax
