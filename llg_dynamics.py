import numpy as np
from scipy.special import jv
from scipy.optimize import brentq
import matplotlib.pyplot as plt 
from mag_params import *
from branch_ranker import DiodeRanker

plt.rcParams['text.usetex'] = True


class LLGDynamics:
    def __init__(
        self,
        mag_params,
        S_mag,
        alpha_loc,
        dt,
        T_eV=0.0,
        eta1=+1,
        eta2=+1,
        B0=0.0,
        omega=0.0,
        drive_amp=1e-9,
        handedness=+1,
        drive_spin=0,
        magnon_convention="phi",
        ramp_time=None,
        ramp_cycles=0.0,
        ramp_type="none",
        seed=None,
    ):
        self.mag_params = mag_params
        self.S_mag = float(S_mag)
        self.alpha_loc = float(alpha_loc)
        self.dt = float(dt)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eta1 = int(eta1)
        self.eta2 = int(eta2)
        self.B0 = float(B0)
        self.T_eV = float(T_eV)
        if self.T_eV < 0.0:
            raise ValueError("T eV must be positive")
        self.omega = float(omega)
        self.drive_amp = float(drive_amp)
        self.handedness = int(handedness)
        self.drive_spin = int(drive_spin)
        self.magnon_convention = magnon_convention
        self.ramp_time = None if ramp_time is None else float(ramp_time)
        self.ramp_cycles = float(ramp_cycles)
        self.ramp_type = str(ramp_type).lower()
        self.R1 = self.branch_rotation(self.eta1)
        self.R2 = self.branch_rotation(self.eta2)
        self.S1_init, self.S2_init = self.branch_initial_state()

    @staticmethod
    def cross_matrix(v):
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ], dtype=float)

    @staticmethod
    def branch_rotation(eta):
        # Rotation such that \eta z = R_k y
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -float(eta)],
            [0.0, float(eta), 0.0],
        ], dtype=float)
    
    # Initialise the spins in either parallel (eta1 = eta2) or antiparallel (eta1 != eta2) state.
    def branch_initial_state(self):
        S1 = np.array([0.0, self.eta1 * self.S_mag, 0.0], dtype=float)
        S2 = np.array([0.0, self.eta2 * self.S_mag, 0.0], dtype=float)
        return S1, S2

    def local_to_lab(self, vec_local, spin_index):
        if spin_index == 0:
            return self.R1.T @ vec_local
        if spin_index == 1:
            return self.R2.T @ vec_local
        raise ValueError("spin_index must be 0 or 1")

    def lab_to_local(self, vec_lab, spin_index):
        if spin_index == 0:
            return self.R1 @ vec_lab
        if spin_index == 1:
            return self.R2 @ vec_lab
        raise ValueError("spin_index must be 0 or 1")

    def drive_period(self):
        if abs(self.omega) == 0.0:
            return np.inf
        return 2.0 * np.pi / abs(self.omega)

    def effective_ramp_time(self):
        if self.ramp_time is not None:
            return self.ramp_time
        return self.ramp_cycles * self.drive_period()

    def drive_envelope(self, t):
        ramp_type = self.ramp_type
        if ramp_type in ("none", "off", "instant", "instantaneous"):
            return 1.0

        t_ramp = self.effective_ramp_time()
        if t_ramp <= 0.0 or not np.isfinite(t_ramp):
            return 1.0
        if t <= 0.0:
            return 0.0

        if ramp_type == "linear":
            return min(t / t_ramp, 1.0)

        if ramp_type == "exponential":
            return 1.0 - np.exp(-t / t_ramp)

        if ramp_type == "cosine":
            if t < t_ramp:
                return 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))
            return 1.0

        raise ValueError("ramp_type must be 'none', 'cosine', 'linear', or 'exponential'")

    def circular_drive_local(self, t):
        envelope = self.drive_envelope(t)
        bx = self.drive_amp * envelope * np.cos(self.omega * t)
        by = self.handedness * self.drive_amp * envelope * np.sin(self.omega * t)
        return np.array([bx, by, 0.0], dtype=float)

    def drive_complex_signal(self, t_vals, spin_index=None):
        if spin_index is None:
            if self.drive_spin in (0,1):
                spin_index = self.drive_spin 
            else:
                spin_index = 0 

        eta = self.eta1 if spin_index == 0 else self.eta2

        t_vals = np.asarray(t_vals, dtype=float)
        b = np.array([self.circular_drive_local(t) for t in t_vals]).reshape((len(t_vals), 3))
        bx = b[:, 0]
        by = b[:, 1]
        if self.magnon_convention == "psi":
            return bx + 1j * eta * by
        if self.magnon_convention == "phi":
            return bx - 1j * eta * by
        raise ValueError("magnon_convention must be 'psi' or 'phi'")

    def B_func(self, t):
        B_static = np.array([0.0, self.B0, 0.0], dtype=float)
        b1 = np.zeros(3)
        b2 = np.zeros(3)

        if self.drive_spin == 0:
            b1 = self.local_to_lab(self.circular_drive_local(t), 0)
        elif self.drive_spin == 1:
            b2 = self.local_to_lab(self.circular_drive_local(t), 1)
        elif self.drive_spin == 2:
            b1 = self.local_to_lab(self.circular_drive_local(t), 0)
            b2 = self.local_to_lab(self.circular_drive_local(t), 1)
        else:
            raise ValueError("drive_spin must be 0, 1, or 2")

        return B_static + b1, B_static + b2

    def A_loc(self):
        return self.alpha_loc * np.eye(3)

    def alpha6(self):
        return self.mag_params.alpha6(self.alpha_loc)

    def generate_noise(self):
        if self.T_eV == 0.0:
            return np.zeros(3), np.zeros(3)

        alpha6 = self.alpha6()
        alpha6 = 0.5 * (alpha6 + alpha6.T)
        eigvals, eigvecs = np.linalg.eigh(alpha6)

        if eigvals[0] < -1e-12:
            raise ValueError(
                "damping covariance is not positive semidefinite: "
                f"minimum eigenvalue = {eigvals[0]}"
            )

        eigvals = np.clip(eigvals, 0.0, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        eta = self.rng.normal(size=6)
        xi = np.sqrt(self.T_eV / self.dt) * (L @ eta)
        return xi[:3], xi[3:]

    def temperature_from_ratio(self, ratio, energy_scale="omega"):
        return float(ratio) * self.temperature_energy_scale_eV(energy_scale)
    def temperature_ratio(self, energy_scale="omega"):
        return self.T_eV / self.temperature_energy_scale_eV(energy_scale)
    def temperature_energy_scale_eV(self, energy_scale="omega"):
        if np.isscalar(energy_scale) and not isinstance(energy_scale, str):
            scale = float(energy_scale)
        else:
            key = str(energy_scale).lower()
            if key in ("omega", "drive", "drive_energy"):
                scale = abs(self.omega)
            elif key in ("branch", "branch_gap", "gap"):
                scale = self.branch_gap_eV()
            elif key in ("b0", "field"):
                scale = abs(self.B0)
            elif key in ("k_scale", "kscale", "interaction"):
                scale = abs(self.mag_params.K_scale)
            elif key in ("exchange", "exchange_norm"):
                scale = np.linalg.norm(self.mag_params.K12)
            else:
                raise ValueError(
                    "energy_scale must be a positive number or one of "
                    "'omega', 'branch_gap', 'B0', 'K_scale', or 'exchange_norm'"
                )

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("the selected temperature energy scale must be finite and positive")
        return scale 

    def normalise_pair(self, S1, S2):
        S1 = self.S_mag * S1 / np.linalg.norm(S1)
        S2 = self.S_mag * S2 / np.linalg.norm(S2)
        return S1, S2

    def effective_fields(self, S1, S2, B_ext1, B_ext2):
        return self.mag_params.effective_fields(S1, S2, B_ext1, B_ext2)
    
    # Damping terms.
    def llg_lhs(self, S1, S2):
        C1 = self.cross_matrix(S1)
        C2 = self.cross_matrix(S2)
        A11 = self.A_loc()
        A22 = self.A_loc()
        A12 = self.mag_params.A12
        A21 = self.mag_params.A21
        M11 = np.eye(3) + C1 @ A11
        M22 = np.eye(3) + C2 @ A22
        M12 = C1 @ A12
        M21 = C2 @ A21
        return np.block([
            [M11, M12],
            [M21, M22],
        ])
    
    # Effective field term.
    def llg_rhs(self, S1, S2, B1, B2, xi1, xi2):
        return np.concatenate([
            np.cross(S1, B1 + xi1),
            np.cross(S2, B2 + xi2),
        ])

    def solve_velocities(self, S1, S2, B1, B2, xi1, xi2):
        M = self.llg_lhs(S1, S2)
        R = self.llg_rhs(S1, S2, B1, B2, xi1, xi2)
        v = np.linalg.solve(M, R)
        return v[:3], v[3:]
    
    # Use a first-order Heun predictor.
    # First take S1 = S1_n, S2 = S2_n. 
    # Then, from \dot{S} = (S_{n+1} - S_{n}) / (t_{n+1} - t_{n}),
    # solve for S_{n+1} = S_{n} + \Delta t \dot{S}, where \dot{S}
    # is the velocity as computed from the LLG and \Delta t = t_{n+1} - t_{n}.
    # Then use this predicted S_{n+1} to compute a second velocity,
    # and return the final velocity as S_{next} = S_n + (S_{n+2} + S_{n+1}) * Delta t / 2 
    def heun_step(self, S1, S2, t):
        xi1, xi2 = self.generate_noise()
        B_ext1, B_ext2 = self.B_func(t)
        B1, B2 = self.effective_fields(S1, S2, B_ext1, B_ext2)
        dS1_1, dS2_1 = self.solve_velocities(S1, S2, B1, B2, xi1, xi2)

        S1_pred = S1 + self.dt * dS1_1
        S2_pred = S2 + self.dt * dS2_1
        # Heun step does not preserve length; renormalise.
        S1_pred, S2_pred = self.normalise_pair(S1_pred, S2_pred)

        B_ext1_pred, B_ext2_pred = self.B_func(t + self.dt)
        B1_pred, B2_pred = self.effective_fields(S1_pred, S2_pred, B_ext1_pred, B_ext2_pred)
        dS1_2, dS2_2 = self.solve_velocities(S1_pred, S2_pred, B1_pred, B2_pred, xi1, xi2)

        S1_next = S1 + 0.5 * self.dt * (dS1_1 + dS1_2)
        S2_next = S2 + 0.5 * self.dt * (dS2_1 + dS2_2)
        S1_next, S2_next = self.normalise_pair(S1_next, S2_next)
        return S1_next, S2_next, B1_pred, B2_pred
    
    # This simple function runs the Heun algorithm for n_steps steps. It will run 
    # from an initial state if provided, or use the initial state given at initialisation time.
    def run(self, n_steps, store_fields=False, initial_state=None, time_offset=0.0):
        if initial_state is None:
            S1 = self.S1_init.copy()
            S2 = self.S2_init.copy()
        else:
            S1 = np.array(initial_state[0], dtype=float, copy=True)
            S2 = np.array(initial_state[1], dtype=float, copy=True)
            S1, S2 = self.normalise_pair(S1, S2)

        trajectory = np.zeros((n_steps, 2, 3))
        fields = np.zeros((n_steps, 2, 3)) if store_fields else None

        for n in range(n_steps):
            t = float(time_offset) + n * self.dt
            S1, S2, B1, B2 = self.heun_step(S1, S2, t)
            trajectory[n, 0] = S1
            trajectory[n, 1] = S2
            if store_fields:
                fields[n, 0] = B1
                fields[n, 1] = B2

        return trajectory, fields
    
    # Compute the circular fields \psi_i^\pm = u_x + \ii u_y.
    # TODO: make branch-aware. 
    def local_magnon_signal(self, trajectory_spin, spin_index):
        R = self.R1 if spin_index == 0 else self.R2
        eta = self.eta1 if spin_index == 0 else self.eta2

        local = trajectory_spin @ R.T
        dx = local[:, 0]
        dy = local[:, 1]
        if self.magnon_convention == "psi":
            return dx + 1j * eta * dy
        if self.magnon_convention == "phi":
            return dx - 1j * eta * dy
        raise ValueError("magnon_convention must be 'psi' or 'phi'")

    @staticmethod
    def complex_amplitude(signal, t_vals, omega, discard_fraction=0.5):
        n0 = int(discard_fraction * len(t_vals))
        t = t_vals[n0:]
        y = signal[n0:]
        phase = np.exp(-1j * omega * t)
        return 2.0 * np.mean(y * phase)

    def measure_response(self, trajectory, discard_fraction=0.5):
        t_vals = self.dt * np.arange(len(trajectory))
        psi1 = self.local_magnon_signal(trajectory[:, 0, :], 0)
        psi2 = self.local_magnon_signal(trajectory[:, 1, :], 1)
        A1 = self.complex_amplitude(psi1, t_vals, self.omega, discard_fraction)
        A2 = self.complex_amplitude(psi2, t_vals, self.omega, discard_fraction)
        return A1, A2

    def torque_check(self):
        S1 = self.S1_init.copy()
        S2 = self.S2_init.copy()
        B_ext1 = np.array([0.0, self.B0, 0.0], dtype=float)
        B_ext2 = np.array([0.0, self.B0, 0.0], dtype=float)
        B1, B2 = self.effective_fields(S1, S2, B_ext1, B_ext2)
        return np.cross(S1, B1), np.cross(S2, B2)
    
    # This uses the helper from the MagneticParams class to determine
    # whether this branch is dynamically stable and the ground state.
    def branch_info(self):
        return self.mag_params.classify_branch(self.B0, self.S_mag, self.eta1, self.eta2)

    def branch_gap_eV(self):
        info = self.branch_info()
        chosen_energy = info["energy"]
        gaps = [
            energy - chosen_energy
            for energy in info["energies"].values()
            if energy - chosen_energy > 1e-15
        ]
        if not gaps:
            return np.inf
        return min(gaps)

    def copy_with(self, **updates):
        params = {
            "mag_params": self.mag_params,
            "S_mag": self.S_mag,
            "alpha_loc": self.alpha_loc,
            "dt": self.dt,
            "seed": self.seed,
            "eta1": self.eta1,
            "eta2": self.eta2,
            "B0": self.B0,
            "T_eV": self.T_eV,
            "omega": self.omega,
            "drive_amp": self.drive_amp,
            "handedness": self.handedness,
            "drive_spin": self.drive_spin,
            "magnon_convention": self.magnon_convention,
            "ramp_time": self.ramp_time,
            "ramp_cycles": self.ramp_cycles,
            "ramp_type": self.ramp_type,
        }
        params.update(updates)
        return LLGDynamics(**params)

    def n_steps_for_cycles(self, n_cycles, dt=None):
        dt_eff = self.dt if dt is None else float(dt)
        return int(np.ceil(float(n_cycles) * self.drive_period() / dt_eff))
    # Run the diode in both directions.
    def run_diode_pair(self, n_steps=None, n_cycles=None, discard_fraction=0.5, store_fields=False, verbose=False):
        if n_steps is None:
            if n_cycles is None:
                raise ValueError("provide either n_steps or n_cycles")
            n_steps = self.n_steps_for_cycles(n_cycles)

        branch_info = self.branch_info()
        outputs = {}

        if verbose:
            print("=" * 70)
            print("Running diode pair")
            print("eta1, eta2:", self.eta1, self.eta2)
            print("B0:", self.B0)
            print("omega:", self.omega)
            print("n_steps:", n_steps)
            print("dt:", self.dt)
            print("period:", self.drive_period())
            print("cycles:", n_steps * self.dt / self.drive_period())
            print("drive_amp:", self.drive_amp)
            print("handedness:", self.handedness)
            print("ramp_cycles:", self.ramp_cycles)
            print("ramp_type:", self.ramp_type)
            print("magnon convention:", self.magnon_convention)
            print("ground:", branch_info["is_ground_state"])
            print("stable:", branch_info["is_locally_stable"])

        for drive_spin in (0, 1):
            dyn = self.copy_with(drive_spin=drive_spin)
            if verbose:
                torque1, torque2 = dyn.torque_check()
                print("-" * 70)
                print("drive spin:", drive_spin + 1)
                print("torque1:", torque1)
                print("torque2:", torque2)
            outputs[drive_spin] = dyn.run_timeseries(
                n_steps=n_steps,
                discard_fraction=discard_fraction,
                store_fields=store_fields,
            )
            if verbose:
                print("response spin 1:", outputs[drive_spin]["abs_A1"])
                print("response spin 2:", outputs[drive_spin]["abs_A2"])

        A21 = outputs[0]["abs_A2"]
        A12 = outputs[1]["abs_A1"]
        eps = 1e-300

        result = {
            "B0": self.B0,
            "omega": self.omega,
            "period": self.drive_period(),
            "branch_info": branch_info,
            "A21": A21,
            "A12": A12,
            "A21_over_A12": A21 / max(A12, eps),
            "A12_over_A21": A12 / max(A21, eps),
            "outputs": outputs,
            "n_steps": n_steps,
            "dt": self.dt,
            "n_cycles": n_steps * self.dt / self.drive_period(),
            "drive_amp": self.drive_amp,
            "handedness": self.handedness,
            "discard_fraction": discard_fraction,
            "ramp_cycles": self.ramp_cycles,
            "ramp_type": self.ramp_type,
            "T_eV": self.T_eV,
            "temperature_ratio_omega": self.temperature_ratio("omega") if abs(self.omega) > 0.0 else np.nan,
        }

        if verbose:
            self.print_diode_result(result)
        return result
    
    def run_timeseries(self, n_steps=None, n_cycles=None, discard_fraction=0.75, store_fields=False):
        if n_steps is None:
            if n_cycles is None:
                raise ValueError("provide either n_steps or n_cycles")
            n_steps = self.n_steps_for_cycles(n_cycles)

        trajectory, fields = self.run(n_steps=n_steps, store_fields=store_fields)
        t = self.dt * np.arange(n_steps)
        psi1 = self.local_magnon_signal(trajectory[:, 0, :], 0)
        psi2 = self.local_magnon_signal(trajectory[:, 1, :], 1)
        A1, A2 = self.measure_response(trajectory, discard_fraction=discard_fraction)
        #A1m, A2m = self.measure_response(trajectory, discard_fraction=discard_fraction, fourier_sign=-1)
 

        return {
            "t": t,
            "trajectory": trajectory,
            "fields": fields,
            "S1": trajectory[:, 0, :],
            "S2": trajectory[:, 1, :],
            "psi1": psi1,
            "psi2": psi2,
            "drive": self.drive_complex_signal(t),
            "A1": A1,
            "A2": A2,
            "abs_A1": abs(A1),
            "abs_A2": abs(A2),
            #"A1_plus": A1,
            #"A2_plus": A2,
            #"abs_A1_plus": abs(A1),
            #"abs_A2_plus": abs(A2),
            #"A1_minus": A1m,
            #"A2_minus": A2m,
            #"abs_A1_minus": abs(A1m),
            #"abs_A2_minus": abs(A2m),
            "n_steps": n_steps,
            "n_cycles": n_steps * self.dt / self.drive_period(),
            "dt": self.dt,
            "drive_spin": self.drive_spin,
            "drive_amp": self.drive_amp,
            "handedness": self.handedness,
            "discard_fraction": discard_fraction,
            "ramp_cycles": self.ramp_cycles,
            "ramp_type": self.ramp_type,
            "T_eV": self.T_eV,
            "temperature_ratio_omega": self.temperature_ratio("omega") if abs(self.omega) > 0.0 else np.nan,
        }    
    @staticmethod
    def print_diode_result(result):
        print("=" * 70)
        keys = [
            "dt",
            "n_cycles",
            "n_steps",
            "drive_amp",
            "handedness",
            "discard_fraction",
            "ramp_cycles",
            "ramp_type",
            "T_eV",
            "temperature_ratio_omega",
            "A21",
            "A12",
            "A21_over_A12",
            "A12_over_A21",
        ]
        for key in keys:
            if key in result:
                print(f"{key}:", result[key])

    @staticmethod
    def print_sweep(results):
        for result in results:
            LLGDynamics.print_diode_result(result)

    def test_drive_amplitude_sweep(
        self,
        amplitudes=(1e-9, 3e-10, 1e-10, 3e-11, 1e-11),
        n_cycles=120,
        discard_fraction=0.75,
        print_results=True,
    ):
        results = []
        for amp in amplitudes:
            result = self.copy_with(drive_amp=amp).run_diode_pair(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
            results.append(result)
        if print_results:
            self.print_sweep(results)
        return results

    def test_timestep_sweep(
        self,
        dts=(1000.0, 500.0, 250.0, 100.0),
        n_cycles=120,
        drive_amp=None,
        discard_fraction=0.75,
        print_results=True,
    ):
        results = []
        amp = self.drive_amp if drive_amp is None else float(drive_amp)
        for dt in dts:
            result = self.copy_with(dt=dt, drive_amp=amp).run_diode_pair(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
            results.append(result)
        if print_results:
            self.print_sweep(results)
        return results

    def test_transient_sweep(
        self,
        n_cycles_values=(40, 80, 120, 200, 400),
        dt=None,
        drive_amp=None,
        discard_fraction=0.75,
        print_results=True,
    ):
        results = []
        dt_eff = self.dt if dt is None else float(dt)
        amp = self.drive_amp if drive_amp is None else float(drive_amp)
        for n_cycles in n_cycles_values:
            result = self.copy_with(dt=dt_eff, drive_amp=amp).run_diode_pair(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
            results.append(result)
        if print_results:
            self.print_sweep(results)
        return results

    def test_discard_fraction_sweep(
        self,
        discard_fractions=(0.25, 0.5, 0.75, 0.875),
        n_cycles=120,
        dt=None,
        drive_amp=None,
        print_results=True,
    ):
        results = []
        dt_eff = self.dt if dt is None else float(dt)
        amp = self.drive_amp if drive_amp is None else float(drive_amp)
        for discard_fraction in discard_fractions:
            result = self.copy_with(dt=dt_eff, drive_amp=amp).run_diode_pair(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
            results.append(result)
        if print_results:
            self.print_sweep(results)
        return results


    def analytic_susceptibility(self, omega_values=None, n_omega=2000, omega_min=None, omega_max=None):
        if omega_values is None:
            omega0 = abs(self.omega)
            if omega_min is None:
                omega_min = 0.5 
            if omega_max is None:
                omega_max = 1.5*omega0 if omega0 > 0.0 else 1.0 
            omega_values = np.linspace(omega_min, omega_max, int(n_omega))
        else:
            omega_values = np.asarray(omega_values, dtype=float)

        c = self.mag_params.ex_and_damp_coeffs()

        J0 = c["J0"]
        J1 = c["J1"]
        D = c["D"]
        a0 = c["a0"]
        a1 = c["a1"]

        a = self.alpha_loc
        S = self.S_mag 
        eta1 = self.eta1
        eta2 = self.eta2 
        B0 = self.B0 

        w = omega_values 
        chi11 = 1j*(w + B0 + eta1*J1*S + 1j*eta2*S*a*w)
        chi22 = 1j*(w + B0 + eta2*J1*S + 1j*eta1*S*a*w)

        chi12_num = eta1 * S * (
            -D + 1j*J0 + a0*w + 1j*a1*w
        )

        chi21_num = eta2 * S * (
            D + 1j*J0 + a0*w - 1j*a1*w
        )

        det_chi_inv = chi11*chi22 - chi12_num*chi21_num 

        chi12 = chi12_num / det_chi_inv
        chi21 = chi21_num / det_chi_inv

        eps = 1e-300 

        abs_chi12 = np.abs(chi12)
        abs_chi21 = np.abs(chi21)

        return {
            "omega": w,
            "num_cycles": w / self.omega,
            "chi12": chi12,
            "chi21": chi21,
            "abs_chi12": abs_chi12,
            "abs_chi21": abs_chi21,
            "det_chi_inv": det_chi_inv,
            "J0": J0,
            "J1": J1,
            "D": D,
            "alpha0": a0,
            "alpha1": a1,
            "alpha": a,
            "S": S,
            "eta1": eta1,
            "eta2": eta2,
            "B0": B0,
        }

    def susceptibility_drive_sweep(
        self,
        omega_values,
        n_cycles=300,
        discard_fraction=0.75,
        drive_amp=None,
    ):
        omega_values = np.asarray(omega_values, dtype=float)
        amp = self.drive_amp if drive_amp is None else float(drive_amp)
    
        chi12 = []
        chi21 = []
    
        for omega in omega_values:
            dyn_base = self.copy_with(
                omega=omega,
                drive_amp=amp,
                ramp_cycles=self.ramp_cycles,
                ramp_type=self.ramp_type,
            )
    
            out_21 = dyn_base.copy_with(drive_spin=0, handedness=self.eta1).run_timeseries(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
            out_12 = dyn_base.copy_with(drive_spin=1, handedness=self.eta2).run_timeseries(
                n_cycles=n_cycles,
                discard_fraction=discard_fraction,
            )
    
            chi21.append(out_21["A2"] / amp)
            chi12.append(out_12["A1"] / amp)
    
        chi12 = np.asarray(chi12)
        chi21 = np.asarray(chi21)
    
        return {
            "omega": omega_values,
            "omega_over_drive": omega_values / self.omega,
            "chi12": chi12,
            "chi21": chi21,
            "abs_chi12": np.abs(chi12),
            "abs_chi21": np.abs(chi21),
            "drive_amp": amp,
            "n_cycles": n_cycles,
            "discard_fraction": discard_fraction,
        }
