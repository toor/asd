import numpy as np 
from scipy.optimize import brentq 
import matplotlib.pyplot as plt
from mag_params import MagneticFormulae, NumericalExchangeTable
from num_exchange import *

class DiodeRanker:
    def __init__(self, K_scale, alpha_scale, S, theta=0.0, x_min=0.1, x_max=50.0, n_x=5000, root_tol=1e-12):
        self.K_scale = float(K_scale)
        self.alpha_scale = float(alpha_scale)
        self.theta = float(theta)
        self.x_max = float(x_max)
        self.x_min = float(x_min) 
        self.n_x = int(n_x)
        self.root_tol = float(root_tol)
        self.S = float(S)
        self._numerical_exchange_cache = {}

    def eV_to_T(self, E_eV, g=2.0):
        mu_B_eV_per_T = 5.788381806e-5
        return E_eV / (g * mu_B_eV_per_T)
    
    
    def eV_to_GHz(self, E_eV):
        h_eV_s = 4.135667696e-15
        return E_eV / h_eV_s / 1e9
    
    def period_from_omega(self, omega):
        if abs(omega) == 0.0:
            return np.inf
        return 2.0 * np.pi / abs(omega)

    def damping_psd_ok(self, mag, tol=1e-12):
        eig0, eigs = mag.damping_psd_info(self.alpha_scale)
        return np.min(eigs) >= -tol, eigs

    def formulae(self,
                 y=None,
                 theta=None,
                 exchange_model="analytic",
                 r_min=None,
                 r_max=None,
                 n_r=None,
                 settings=None,
    ):
        if y is None:
            raise ValueError("y must be provided")
        if theta is None:
            theta = self.theta
        if exchange_model == "analytic":
            return MagneticFormulae(y=y, theta=theta)
        if exchange_model == "numerical":
            return self.numerical_formulae(
                y=y,
                theta=theta,
                r_min=r_min,
                r_max=r_max,
                n_r=n_r,
                settings=settings
            )

    def numerical_formulae(self, y, theta, r_min, r_max, n_r, settings):
        default_settings = self.default_numerical_exchange_settings(y)
        user_settings = {} if settings is None else dict(settings)

        merged_settings = {
            **default_settings,
            **user_settings,
        }

        key = (
            float(y),
            float(theta),
            float(r_min),
            float(r_max),
            float(n_r),
            tuple(sorted(merged_settings.items())),
        )

        if key not in self._numerical_exchange_cache:
            xs = np.linspace(r_min, r_max, int(n_r))

            values = scan_components(
                ("xx", "yy", "zz", "xz", "yz", "xy"),
                xs,
                y,
                theta=theta,
                **merged_settings
            )

            self._numerical_exchange_cache[key] = NumericalExchangeTable(
                y=y,
                theta=theta,
                xs=xs,
                values=values,
            )

        return self._numerical_exchange_cache[key]

    def default_numerical_exchange_settings(self, y):
        y = float(y)

        d_plus = 0.5 * (1 - y**2)

        rashba_gap = max(1.0 - y, 1e-3)

        settings = {
            "e_max": 500.0,
            "n_e": 900,
            "n_h": 160,
            "n_rho": 700,
            "rho_scale": 1.0,
        }

        if y > 0.5:
            settings.update({
                "n_e": 1100,
                "n_h": 200,
                "n_rho": 900,
            })
        if y > 0.75:
            settings.update({
                "e_max": 700.0,
                "n_e": 1400,
                "n_h": 260,
                "n_rho": 1100,
                "rho_scale": 0.5,
            })

        if d_plus < 0.05:
            settings.update({
                "n_h": max(settings["n_h"], 320),
                "n_rho": max(settings["n_rho"], 1300),
                "rho_scale": min(settings["rho_scale"], 0.35),
            })

        return settings

    def make_params(self, formulae, r):
        return formulae.make_params(
            x=r,
            alpha_scale=self.alpha_scale,
            K_scale=self.K_scale,
        )

    def time_resolution_info(
        self,
        omega,
        dt=None,
        n_cycles=40,
        min_steps_per_period=50,
        max_n_steps=2_000_000,
    ):
        omega_abs = abs(float(omega))
    
        if not np.isfinite(omega_abs) or omega_abs <= 0.0:
            return {
                "period": np.inf,
                "dt": np.nan,
                "dt_recommended": np.nan,
                "steps_per_period": np.nan,
                "recommended_n_steps": np.inf,
                "time_resolved": False,
                "time_failure_reason": "nonfinite_or_zero_omega",
            }
    
        period = 2.0 * np.pi / omega_abs
    
        if dt is None:
            dt_used = period / min_steps_per_period
        else:
            dt_used = float(dt)
    
        steps_per_period = period / dt_used
        recommended_n_steps = int(np.ceil(n_cycles * steps_per_period))
    
        if steps_per_period < min_steps_per_period:
            time_resolved = False
            reason = "time_resolution_too_low"
        elif recommended_n_steps > max_n_steps:
            time_resolved = False
            reason = "too_many_time_steps"
        else:
            time_resolved = True
            reason = ""
    
        return {
            "period": period,
            "dt": dt_used,
            "dt_recommended": period / min_steps_per_period,
            "steps_per_period": steps_per_period,
            "recommended_n_steps": recommended_n_steps,
            "time_resolved": time_resolved,
            "time_failure_reason": reason,
        }
    
    def add_time_info(
        self,
        row,
        omega,
        dt,
        n_cycles,
        min_steps_per_period,
        max_n_steps,
    ):
        time_info = self.time_resolution_info(
            omega=omega,
            dt=dt,
            n_cycles=n_cycles,
            min_steps_per_period=min_steps_per_period,
            max_n_steps=max_n_steps,
        )
    
        if not time_info["time_resolved"]:
            return None, time_info["time_failure_reason"], time_info
    
        return {
            **row,
            **time_info,
        }, None, time_info    


    def find_reduced_diode_roots(self, formulae, x_max=None, x_min=None, n_x=None):
        if x_min is None:
            x_min = self.x_min 
        if x_max is None:
            x_max = self.x_max 
        if n_x is None:
            n_x = self.n_x
        x_vals = np.linspace(x_min, x_max, int(n_x))
        
        vals = np.array([
            formulae.diode_condition(x)
            for x in x_vals
        ])
    
        roots = []
    
        for i in range(len(x_vals) - 1):
            x1 = x_vals[i]
            x2 = x_vals[i + 1]
    
            f1 = vals[i]
            f2 = vals[i + 1]
    
            if not np.isfinite(f1) or not np.isfinite(f2):
                continue
            
            # First check if the point itself is a root, up to a small 
            # numerical tolerance
            if abs(f1) < self.root_tol:
                roots.append(x1)
                continue
            
            # f changes sign between i and i + 1, indicating a root 
            # which is not captured by the current sample spacing.
            if f1 * f2 < 0.0:
                try:
                    root = brentq(
                        formulae.diode_condition,
                        x1,
                        x2,
                        xtol=1e-13,
                        rtol=1e-13,
                        maxiter=100,
                    )
                    roots.append(root)
                except ValueError:
                    pass
    
        if not roots:
            return np.array([]), x_vals, vals
    
        roots = np.array(sorted(roots))
        
        # Remove duplicate roots
        deduped = [roots[0]]
        for r in roots[1:]:
            if abs(r - deduped[-1]) > 1e-8:
                deduped.append(r)
    
        return np.array(deduped), x_vals, vals 

    def branch_allowed(self, eta1, eta2, branch_type="parallel"):
        if branch_type == "parallel":
            return eta1 == eta2 
        
        if branch_type == "antiparallel":
            return eta1 != eta2 
        
        if branch_type == "all":
            return True 
        raise ValueError("branch_type must be 'parallel', 'antiparallel', or 'all'")

    def branch_list(self, branch_type="parallel"):
        if branch_type == "parallel":
            return [(+1, +1), (-1, -1)]
        if branch_type == "antiparallel":
            return [(+1, -1), (-1, +1)]
        if branch_type == "all":
            return [(+1, +1), (-1, -1), (+1, -1), (-1, +1)]

    def exact_branch_rows(self, mag, branch_type="parallel"):
        rows = []
    
        for eta1, eta2 in self.branch_list(branch_type):
            for root_sign in (-1, +1):
                for suppress_direction in ("12", "21"):
                    try:
                        B0 = mag.candidate_B0(
                            eta1=eta1,
                            eta2=eta2,
                            root_sign=root_sign,
                            suppress_direction=suppress_direction,
                            S=self.S,
                        )
    
                        omega = mag.branch_resonance(
                            B0=B0,
                            eta1=eta1,
                            eta2=eta2,
                            root_sign=root_sign,
                            S=self.S,
                        )
    
                        info = mag.classify_branch(B0, self.S, eta1, eta2)
    
                        rows.append({
                            "branch_mode": "exact_root",
                            "eta1": eta1,
                            "eta2": eta2,
                            "root_sign": root_sign,
                            "suppress_direction": suppress_direction,
                            "B0": B0,
                            "omega": omega,
                            "is_locally_stable": info["is_locally_stable"],
                            "is_ground_state": info["is_ground_state"],
                            "stab1": info["stab1"],
                            "stab2": info["stab2"],
                            "energy": info["energy"],
                        })
    
                    except ValueError as err:
                        rows.append({
                            "branch_mode": "exact_root",
                            "eta1": eta1,
                            "eta2": eta2,
                            "root_sign": root_sign,
                            "suppress_direction": suppress_direction,
                            "error": str(err),
                        })
    
        return rows 

    def contrast_branch_rows(self, mag, branch_type="parallel"):
        rows = []
    
        for eta1, eta2 in self.branch_list(branch_type):
            for root_sign in (-1, +1):
                for suppress_direction in ("12", "21"):
                    try:
                        contrast = mag.weak_damping_contrast_at_optimal_field(
                            eta1=eta1,
                            eta2=eta2,
                            root_sign=root_sign,
                            suppress_direction=suppress_direction,
                            S=self.S,
                        )
    
                        info = mag.classify_branch(
                            B0=contrast["B0"],
                            S=self.S,
                            eta1=eta1,
                            eta2=eta2,
                        )
    
                        rows.append({
                            **contrast,
                            "branch_mode": "contrast_optimised",
                            "eta1": eta1,
                            "eta2": eta2,
                            "root_sign": root_sign,
                            "suppress_direction": suppress_direction,
                            "is_locally_stable": info["is_locally_stable"],
                            "is_ground_state": info["is_ground_state"],
                            "stab1": info["stab1"],
                            "stab2": info["stab2"],
                            "energy": info["energy"],
                        })
    
                    except ValueError as err:
                        rows.append({
                            "branch_mode": "contrast_optimised",
                            "eta1": eta1,
                            "eta2": eta2,
                            "root_sign": root_sign,
                            "suppress_direction": suppress_direction,
                            "error": str(err),
                        })
    
        return rows
    # Computes, for all energy branches, roots, and diode directions,
    def record_rejection(self, diagnostics, stage, reason, **data):
        if diagnostics is None:
            return 
        diagnostics.setdefault(stage, []).append({
            "reason": reason,
            **data,
        })
    
    def diode_scan_summary(self, formulae, r_grid, vals):
        vals = np.asarray(vals, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)

        finite = np.isfinite(vals)

        out = {
            "y": float(formulae.y),
            "theta": float(formulae.theta),
            "n_grid": int(len(r_grid)),
            "n_finite": int(np.sum(finite)),
            "n_nonfinite": int(np.sum(~finite)),
        }

        if not np.any(finite):
            out.update({
                "min_abs_diode_condition": np.inf,
                "best_r": np.nan,
                "best_value": np.nan,
                "value_min": np.nan,
                "value_max": np.nan,
                "n_sign_changes": 0,
            })

            return out 

        finite_r = r_grid[finite]
        finite_v = vals[finite]

        i_best = int(np.argmin(np.abs(finite_v)))

        sign_changes = np.sum(
            np.isfinite(finite_v[:-1])
            & np.isfinite(finite_v[1:])
            & (finite_v[:-1] * finite_v[1:] < 0.0)
        )
        out.update({
            "min_abs_diode_condition": float(abs(finite_v[i_best])),
            "best_r": float(finite_r[i_best]),
            "best_value": float(finite_v[i_best]),
            "value_min": float(np.min(finite_v)),
            "value_max": float(np.max(finite_v)),
            "n_sign_changes": int(sign_changes),
        })

        return out
    
    def branch_failure_reason(
        self,
        br,
        branch_type="parallel",
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        min_isolation=None,
    ):

        if not self.branch_allowed(br["eta1"], br["eta2"], branch_type):
            return "wrong_branch_type", {
                "branch_type": branch_type,
            }

        if "error" in br:
            return "branch_formula_error", {"branch_error": br["error"]}
        
        if not br["is_locally_stable"]:
            return "locally_unstable", {
                "stab1": br.get("stab1", np.nan),
                "stab2": br.get("stab2", np.nan),
            }
    
        if not br["is_ground_state"]:
            return "not_ground_state", {
                "energy": br.get("energy", np.nan),
            }
    
        B0_abs = abs(br["B0"])
        omega_abs = abs(br["omega"])
    
        if B0_abs_min is not None and B0_abs < B0_abs_min:
            return "B0_below_min", {"B0_abs": B0_abs, "B0_abs_min": B0_abs_min}
    
        if B0_abs_max is not None and B0_abs > B0_abs_max:
            return "B0_above_max", {"B0_abs": B0_abs, "B0_abs_max": B0_abs_max}
    
        if omega_abs_min is not None and omega_abs < omega_abs_min:
            return "omega_below_min", {
                "omega_abs": omega_abs,
                "omega_abs_min": omega_abs_min,
            }
    
        if omega_abs_max is not None and omega_abs > omega_abs_max:
            return "omega_above_max", {
                "omega_abs": omega_abs,
                "omega_abs_max": omega_abs_max,
            }
    
        if min_isolation is not None:
            if "isolation" not in br:
                return "missing_isolation", {}
            if br["isolation"] < min_isolation:
                return "isolation_below_min", {
                    "isolation": br["isolation"],
                    "min_isolation": min_isolation,
                }
    
        return None, {}

    def valid_branch_rows(
        self,
        branch_rows,
        branch_type="parallel",
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        min_isolation=None,
        diagnostics=None,
        context=None,
    ):
        rows = []
        context = {} if context is None else dict(context)
    
        for br in branch_rows:
            reason, extra = self.branch_failure_reason(
                br,
                branch_type=branch_type,
                B0_abs_min=B0_abs_min,
                B0_abs_max=B0_abs_max,
                omega_abs_min=omega_abs_min,
                omega_abs_max=omega_abs_max,
                min_isolation=min_isolation,
            )
    
            if reason is not None:
                self.record_rejection(
                    diagnostics,
                    stage="branch",
                    reason=reason,
                    **context,
                    eta1=br.get("eta1"),
                    eta2=br.get("eta2"),
                    root_sign=br.get("root_sign"),
                    suppress_direction=br.get("suppress_direction"),
                    B0=br.get("B0", np.nan),
                    omega=br.get("omega", np.nan),
                    **extra,
                )
                continue
    
            rows.append(br)
    
        return rows    

    def root_quality(
        self,
        r,
        formulae,
        kF_inv_angstrom=None,
        kF_min_inv_angstrom=None,
        kF_max_inv_angstrom=None,
        R_min_angstrom=None,
        R_max_angstrom=None,
    ):
        mag = self.make_params(formulae, r)
    
        c_full = mag.ex_and_damp_coeffs()
        Omega0, Omega1, Omega1_sq = mag.energy_gaps()
    
        J0_re = formulae.J_xx(r)
        D_re = formulae.J_xz(r)
    
        a0_re = formulae.A_xx(r) / 4.0
        a1_re = formulae.A_xz(r) / 4.0
    
        term_D_re = D_re * a1_re
        term_J_re = J0_re * a0_re
    
        reduced_metric = term_D_re + term_J_re
        reduced_denom = max(abs(term_D_re), abs(term_J_re), 1e-300)
        relative_reduced_metric = abs(reduced_metric) / reduced_denom
    
        row = {
            "r": r,
            "y": formulae.y,
            "theta": formulae.theta,
    
            "F": formulae.F(r),
            "exchange_norm": np.linalg.norm(mag.K12),
    
            "J0": c_full["J0"],
            "J1": c_full["J1"],
            "D": c_full["D"],
            "a0": c_full["a0"],
            "a1": c_full["a1"],
    
            "Omega0": Omega0,
            "Omega1": Omega1,
            "Omega1_sq": Omega1_sq,
    
            "term_D_reduced": term_D_re,
            "term_J_reduced": term_J_re,
            "reduced_diode_metric": reduced_metric,
            "relative_reduced_diode_metric": relative_reduced_metric,
            "full_diode_metric": mag.diode_metric(),
        }
    
        if kF_inv_angstrom is not None:
            kF_inv_angstrom = float(kF_inv_angstrom)
    
            row["kF_inv_angstrom"] = kF_inv_angstrom
            row["qR_inv_angstrom"] = formulae.y * kF_inv_angstrom
            row["R_angstrom"] = r / kF_inv_angstrom
            row["R_nm"] = row["R_angstrom"] / 10.0
    
        has_kF_window = all(
            value is not None
            for value in (
                kF_min_inv_angstrom,
                kF_max_inv_angstrom,
                R_min_angstrom,
                R_max_angstrom,
            )
        )
    
        if has_kF_window:
            kF_min_inv_angstrom = float(kF_min_inv_angstrom)
            kF_max_inv_angstrom = float(kF_max_inv_angstrom)
            R_min_angstrom = float(R_min_angstrom)
            R_max_angstrom = float(R_max_angstrom)
    
            kF_required_min = r / R_max_angstrom
            kF_required_max = r / R_min_angstrom
    
            kF_allowed_min = max(kF_min_inv_angstrom, kF_required_min)
            kF_allowed_max = min(kF_max_inv_angstrom, kF_required_max)
    
            physically_realisable = kF_allowed_min <= kF_allowed_max
    
            row.update({
                "kF_required_min_inv_angstrom": kF_required_min,
                "kF_required_max_inv_angstrom": kF_required_max,
                "kF_allowed_min_inv_angstrom": kF_allowed_min,
                "kF_allowed_max_inv_angstrom": kF_allowed_max,
                "kF_window_width_inv_angstrom": (
                    kF_allowed_max - kF_allowed_min
                    if physically_realisable else 0.0
                ),
                "physically_realisable": physically_realisable,
            })
    
            if physically_realisable:
                kF_best_inv_angstrom = np.sqrt(kF_allowed_min * kF_allowed_max)
                R_best_angstrom = r / kF_best_inv_angstrom
    
                row.update({
                    "kF_best_inv_angstrom": kF_best_inv_angstrom,
                    "qR_best_inv_angstrom": formulae.y * kF_best_inv_angstrom,
                    "R_best_angstrom": R_best_angstrom,
                    "R_best_nm": R_best_angstrom / 10.0,
                })
            else:
                row.update({
                    "kF_best_inv_angstrom": np.nan,
                    "qR_best_inv_angstrom": np.nan,
                    "R_best_angstrom": np.nan,
                    "R_best_nm": np.nan,
                })
    
        return row
    
    def rank_geometry_candidates(
        self,
        geometry_rows,
        branch_row_func,
        sorter,
        score_func=None,
        branch_type="parallel",
        F_min=1e-5,
        exchange_min=None,
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        min_isolation=None,
        dt=1e3,
        n_cycles=40,
        min_steps_per_period=50,
        max_n_steps=2_000_000,
        diagnostics=None,
    ):
        rows = []
    
        for geo in geometry_rows:
            formulae = geo["formulae"]
            r = geo["r"]
            rq = geo["quality"]
            metadata = geo.get("metadata", {})
    
            context = {
                **metadata,
                "y": rq.get("y", formulae.y),
                "r": r,
                "J0": rq.get("J0", np.nan),
                "J1": rq.get("J1", np.nan),
                "D": rq.get("D", np.nan),
                "a0": rq.get("a0", np.nan),
                "a1": rq.get("a1", np.nan),
                "F": rq.get("F", np.nan),
                "exchange_norm": rq.get("exchange_norm", np.nan),
                "relative_reduced_diode_metric": rq.get(
                    "relative_reduced_diode_metric", np.nan
                ),
                "full_diode_metric": rq.get("full_diode_metric", np.nan),
            }
    
            if abs(rq["F"]) < F_min:
                self.record_rejection(
                    diagnostics,
                    "geometry",
                    "F_below_min",
                    **context,
                    F_min=F_min,
                )
                continue
    
            if exchange_min is not None and rq["exchange_norm"] < exchange_min:
                self.record_rejection(
                    diagnostics,
                    "geometry",
                    "exchange_norm_below_min",
                    **context,
                    exchange_min=exchange_min,
                )
                continue
    
            mag = self.make_params(formulae, r)
    
            psd_ok, psd_eigs = self.damping_psd_ok(mag)
            if not psd_ok:
                self.record_rejection(
                    diagnostics,
                    "geometry",
                    "damping_not_psd",
                    **context,
                    alpha6_min_eig=float(np.min(psd_eigs)),
                )
                continue
    
            branch_rows = branch_row_func(mag, branch_type=branch_type)
    
            branch_rows = self.valid_branch_rows(
                branch_rows=branch_rows,
                branch_type=branch_type,
                B0_abs_min=B0_abs_min,
                B0_abs_max=B0_abs_max,
                omega_abs_min=omega_abs_min,
                omega_abs_max=omega_abs_max,
                min_isolation=min_isolation,
                diagnostics=diagnostics,
                context=context,
            )
    
            if len(branch_rows) == 0:
                self.record_rejection(
                    diagnostics,
                    "geometry",
                    "no_valid_branch_rows",
                    **context,
                )
                continue
    
            for br in branch_rows:
                score_data = {}
                if score_func is not None:
                    score_data = score_func(br)
    
                row = {
                    **rq,
                    **metadata,
                    **br,
                    "alpha6_eigs": psd_eigs,
                    **score_data,
                }
    
                row, reason, time_info = self.add_time_info(
                    row=row,
                    omega=br["omega"],
                    dt=dt,
                    n_cycles=n_cycles,
                    min_steps_per_period=min_steps_per_period,
                    max_n_steps=max_n_steps,
                )
    
                if reason is not None:
                    self.record_rejection(
                        diagnostics,
                        "time",
                        reason,
                        **context,
                        eta1=br.get("eta1"),
                        eta2=br.get("eta2"),
                        root_sign=br.get("root_sign"),
                        suppress_direction=br.get("suppress_direction"),
                        B0=br.get("B0", np.nan),
                        omega=br.get("omega", np.nan),
                        **time_info,
                    )
                    continue
    
                rows.append(row)
    
        return sorter(rows)

    def material_contrast_score(self, branch_row, allowed_weight=0.0):
        eps = 1e-300
        score = (
            np.log10(max(branch_row["isolation"], eps))
            + float(allowed_weight) * np.log10(max(branch_row["allowed_abs"], eps))
        )
        return {"score": score}

    #def sort_exact_root_rows(self, rows):
    #    diode_metric_floor = 1e-10 

    #    rows.sort(
    #        key=lambda row: (
    #            max(row["relative_reduced_diode_metric"], diode_metric_floor),
    #            -row["exchange_norm"],
    #            row.get("R_best_angstrom", row.get("R_angstrom", np.inf)),
    #            abs(row["B0"]),
    #            row["recommended_n_steps"]
    #        )
    #    )

    #    return rows
    
    def sort_exact_root_rows(self, rows):
        return sorted(
            rows,
            key=lambda row: (
                abs(row["B0"]),                      # prefer lower field
                abs(row["omega"]),                   # then lower drive frequency
                row.get("relative_reduced_diode_metric", np.inf),
                -abs(row.get("F", 0.0)),             # then stronger geometric factor
                -row.get("exchange_norm", 0.0),      # then stronger exchange scale
            ),
        )    
    
    def sort_material_contrast_rows(self, rows):
        rows.sort(
            key=lambda row: (
                -row["score"],
                -row["isolation"],
                -row["allowed_abs"],
                row["R_angstrom"],
                abs(row["B0"]),
                row["recommended_n_steps"],
            )
        )
        return rows

    def rank_fixed_y_diode_candidates(
        self,
        y,
        kF_inv_angstrom,
        theta=None,
        R_min_angstrom=3.0,
        R_max_angstrom=50.0,
        F_min=1e-5,
        exchange_min=None,
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        branch_type="parallel",
        dt=1e3,
        n_cycles=40,
        min_steps_per_period=50,
        max_n_steps=2_000_000,
    ):
        formulae = self.formulae(y=y, theta=theta)
        roots, _, _ = self.find_reduced_diode_roots(formulae)
    
        geometry_rows = []
    
        for r in roots:
            rq = self.root_quality(
                formulae=formulae,
                r=r,
                kF_inv_angstrom=kF_inv_angstrom,
            )
    
            if not (R_min_angstrom <= rq["R_angstrom"] <= R_max_angstrom):
                continue
    
            geometry_rows.append({
                "formulae": formulae,
                "r": r,
                "quality": rq,
            })
    
        return self.rank_geometry_candidates(
            geometry_rows=geometry_rows,
            branch_row_func=self.exact_branch_rows,
            sorter=self.sort_exact_root_rows,
            score_func=None,
            branch_type=branch_type,
            F_min=F_min,
            exchange_min=exchange_min,
            B0_abs_min=B0_abs_min,
            B0_abs_max=B0_abs_max,
            omega_abs_min=omega_abs_min,
            omega_abs_max=omega_abs_max,
            min_isolation=None,
            dt=dt,
            n_cycles=n_cycles,
            min_steps_per_period=min_steps_per_period,
            max_n_steps=max_n_steps,
        )

    def rank_dimensionless_diode_candidates(
        self,
        y_values,
        theta=None,
        r_min=0.1,
        r_max=50.0,
        n_r=6000,
        kF_min_inv_angstrom=0.02,
        kF_max_inv_angstrom=0.30,
        kF_window_min_inv_angstrom=0.01,
        R_min_angstrom=3.0,
        R_max_angstrom=100.0,
        F_min=1e-5,
        exchange_min=None,
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        branch_type="parallel",
        dt=1e3,
        n_cycles=40,
        min_steps_per_period=50,
        max_n_steps=2_000_000,
        return_diagnostics=False,
        exchange_model="analytic",
        numerical_exchange_settings=None,
    ):
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "root_scan": [],
                "geometry": [],
                "branch": [],
                "time": [],
            }
    
        geometry_rows = []
        total_roots = 0
        
        print("RANKING DIODE CANDIDATES IN (r, y) PHASE SPACE")
        print(f"y_min = {np.min(y_values)}; y_max = {np.max(y_values)}")
        print(f"r_min = {r_min}; r_max={r_max}")
        print(f"kF_min = {kF_min_inv_angstrom}; kF_max = {kF_max_inv_angstrom}")
        print(f"kF_window_min = {kF_window_min_inv_angstrom}")
        print(f"F_min = {F_min}")
        print(f"B0_abs_min = {B0_abs_min if B0_abs_min is not None else 0.0}")
        print(f"B0_abs_max = {B0_abs_max if B0_abs_max is not None else 0.0}")
        print(f"omega_abs_min = {omega_abs_min if omega_abs_min is not None else 0.0}")
        print(f"omega_abs_max = {omega_abs_max if omega_abs_max is not None else 0.0}")

        for y in y_values:
            formulae = self.formulae(
                y=y,
                theta=theta,
                exchange_model=exchange_model,
                r_min=r_min,
                r_max=r_max,
                n_r=n_r,
                settings=numerical_exchange_settings,
            )
    
            roots, r_grid, vals = self.find_reduced_diode_roots(
                formulae=formulae,
                x_min=r_min,
                x_max=r_max,
                n_x=n_r,
            )
    
            if diagnostics is not None:
                scan = self.diode_scan_summary(formulae, r_grid, vals)
                scan["n_roots"] = int(len(roots))
                scan["both_rashba_contours_exist"] = bool(y < 1.0)
                diagnostics["root_scan"].append(scan)
    
            total_roots += len(roots)
    
            if len(roots) == 0:
                self.record_rejection(
                    diagnostics,
                    "geometry",
                    "no_reduced_diode_roots_for_y",
                    y=float(y),
                )
                continue
    
            for r in roots:
                rq = self.root_quality(
                    formulae=formulae,
                    r=r,
                    kF_min_inv_angstrom=kF_min_inv_angstrom,
                    kF_max_inv_angstrom=kF_max_inv_angstrom,
                    R_min_angstrom=R_min_angstrom,
                    R_max_angstrom=R_max_angstrom,
                )
    
                base = {
                    "y": rq["y"],
                    "r": rq["r"],
                    "J0": rq["J0"],
                    "J1": rq["J1"],
                    "D": rq["D"],
                    "a0": rq["a0"],
                    "a1": rq["a1"],
                    "F": rq["F"],
                    "exchange_norm": rq["exchange_norm"],
                    "relative_reduced_diode_metric": rq["relative_reduced_diode_metric"],
                    "full_diode_metric": rq["full_diode_metric"],
                    "kF_allowed_min_inv_angstrom": rq.get(
                        "kF_allowed_min_inv_angstrom", np.nan
                    ),
                    "kF_allowed_max_inv_angstrom": rq.get(
                        "kF_allowed_max_inv_angstrom", np.nan
                    ),
                }
    
                if not rq["physically_realisable"]:
                    self.record_rejection(
                        diagnostics,
                        "geometry",
                        "no_kF_R_window",
                        **base,
                    )
                    continue
    
                if rq["kF_window_width_inv_angstrom"] < kF_window_min_inv_angstrom:
                    self.record_rejection(
                        diagnostics,
                        "geometry",
                        "kF_window_too_narrow",
                        **base,
                        kF_window_width_inv_angstrom=rq[
                            "kF_window_width_inv_angstrom"
                        ],
                        kF_window_min_inv_angstrom=kF_window_min_inv_angstrom,
                    )
                    continue
    
                geometry_rows.append({
                    "formulae": formulae,
                    "r": r,
                    "quality": rq,
                })
    
        rows = self.rank_geometry_candidates(
            geometry_rows=geometry_rows,
            branch_row_func=self.exact_branch_rows,
            sorter=self.sort_exact_root_rows,
            score_func=None,
            branch_type=branch_type,
            F_min=F_min,
            exchange_min=exchange_min,
            B0_abs_min=B0_abs_min,
            B0_abs_max=B0_abs_max,
            omega_abs_min=omega_abs_min,
            omega_abs_max=omega_abs_max,
            min_isolation=None,
            dt=dt,
            n_cycles=n_cycles,
            min_steps_per_period=min_steps_per_period,
            max_n_steps=max_n_steps,
            diagnostics=diagnostics,
        )
    
        if return_diagnostics:
            return rows, total_roots, diagnostics
    
        return rows, total_roots
    
    def material_parameters_from_row(self, material):
        name = material.get("name", "unnamed material")
    
        kF = (
            material.get("kF_inv_angstrom")
            if "kF_inv_angstrom" in material
            else material.get("kF_Ainv")
        )
        if kF is None:
            raise ValueError(f"material {name!r} must define kF_inv_angstrom or kF_Ainv")
        kF = float(kF)
    
        if "qR_inv_angstrom" in material:
            qR = float(material["qR_inv_angstrom"])
            y = qR / kF
        elif "qR_Ainv" in material:
            qR = float(material["qR_Ainv"])
            y = qR / kF
        elif "y" in material:
            y = float(material["y"])
            qR = y * kF
        else:
            raise ValueError(f"material {name!r} must define qR_inv_angstrom, qR_Ainv, or y")
    
        if kF <= 0.0:
            raise ValueError(f"material {name!r} has non-positive kF")
        if y <= 0.0:
            raise ValueError(f"material {name!r} has non-positive y")
    
        return name, kF, qR, y

    def rank_material_contrast_candidates(
        self,
        materials,
        theta=None,
        R_min_angstrom=3.0,
        R_max_angstrom=100.0,
        n_R=1000,
        R_values_angstrom=None,
        F_min=1e-5,
        exchange_min=None,
        B0_abs_min=None,
        B0_abs_max=None,
        omega_abs_min=None,
        omega_abs_max=None,
        min_isolation=None,
        branch_type="parallel",
        dt=1e3,
        n_cycles=40,
        min_steps_per_period=50,
        max_n_steps=2_000_000,
        allowed_weight=0.0,
    ):
        if R_values_angstrom is None:
            R_values = np.linspace(R_min_angstrom, R_max_angstrom, int(n_R))
        else:
            R_values = np.asarray(R_values_angstrom, dtype=float)
    
        geometry_rows = []
    
        for material in materials:
            material_name, kF_inv_angstrom, qR_inv_angstrom, y = (
                self.material_parameters_from_row(material)
            )
    
            formulae = self.formulae(y=y, theta=theta)
    
            for R_angstrom in R_values:
                if R_angstrom <= 0.0:
                    continue
    
                r = kF_inv_angstrom * R_angstrom
    
                rq = self.root_quality(
                    formulae=formulae,
                    r=r,
                    kF_inv_angstrom=kF_inv_angstrom,
                )
    
                geometry_rows.append({
                    "formulae": formulae,
                    "r": r,
                    "quality": rq,
                    "metadata": {
                        "material": material_name,
                        "kF_inv_angstrom": kF_inv_angstrom,
                        "qR_inv_angstrom": qR_inv_angstrom,
                    },
                })
    
        return self.rank_geometry_candidates(
            geometry_rows=geometry_rows,
            branch_row_func=self.contrast_branch_rows,
            sorter=self.sort_material_contrast_rows,
            score_func=lambda br: self.material_contrast_score(
                br,
                allowed_weight=allowed_weight,
            ),
            branch_type=branch_type,
            F_min=F_min,
            exchange_min=exchange_min,
            B0_abs_min=B0_abs_min,
            B0_abs_max=B0_abs_max,
            omega_abs_min=omega_abs_min,
            omega_abs_max=omega_abs_max,
            min_isolation=min_isolation,
            dt=dt,
            n_cycles=n_cycles,
            min_steps_per_period=min_steps_per_period,
            max_n_steps=max_n_steps,
        )  
    
    def print_material_contrast_candidate(self, row):
        print("=" * 70)
        print("material:", row["material"])
        print("kF [1/angstrom]:", row["kF_inv_angstrom"])
        print("qR [1/angstrom]:", row["qR_inv_angstrom"])
        print("y = qR/kF:", row["y"])
        print("R [nm]:", row["R_nm"])
        print("r = kF R:", row["r"])
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
        print("abs_chi12:", row["abs_chi12"])
        print("abs_chi21:", row["abs_chi21"])
        print("allowed_abs:", row["allowed_abs"])
        print("suppressed_abs:", row["suppressed_abs"])
        print("isolation:", row["isolation"])
        print("isolation [dB]:", row["isolation_dB"])
        print("score:", row["score"])
    
        B0_eV = row["B0"]
        omega_eV = abs(row["omega"])
        print("B0 [eV]:", B0_eV)
        print("B0 [meV]:", 1e3 * B0_eV)
        print("B0 [T]:", self.eV_to_T(B0_eV))
        print("omega [eV]:", omega_eV)
        print("omega [meV]:", 1e3 * omega_eV)
        print("omega [GHz]:", self.eV_to_GHz(omega_eV))
    
    
    def plot_material_contrast_scan(self, rows, filename=None, title=None):
        if not rows:
            raise ValueError("no material-constrained contrast rows to plot")
    
        materials = list(dict.fromkeys(row["material"] for row in rows))
        fig = plt.figure(figsize=(7.0, 5.2))
        ax_iso = fig.add_subplot(2, 1, 1)
        ax_amp = fig.add_subplot(2, 1, 2, sharex=ax_iso)
    
        for material in materials:
            subset = [row for row in rows if row["material"] == material]
            x = np.asarray([row["R_nm"] for row in subset], dtype=float)
            isolation_dB = np.asarray([row["isolation_dB"] for row in subset], dtype=float)
            allowed = np.asarray([row["allowed_abs"] for row in subset], dtype=float)
    
            order = np.argsort(x)
            ax_iso.scatter(x[order], isolation_dB[order], s=12, label=material)
            ax_amp.scatter(x[order], allowed[order], s=12, label=material)
    
        ax_iso.set_ylabel("isolation [dB]")
        ax_iso.legend(frameon=False)
    
        ax_amp.set_yscale("log")
        ax_amp.set_ylabel("allowed response")
        ax_amp.set_xlabel(r"$R$ [nm]")
    
        if title is not None:
            fig.suptitle(title)
    
        fig.tight_layout()
    
        if filename is not None:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
    
        return fig, (ax_iso, ax_amp)

    def print_examples(self, items, reason, keys, n=8):
        examples = [x for x in items if x.get("reason") == reason]
        if not examples:
            return
    
        def sort_key(x):
            if reason == "B0_above_max":
                return abs(x.get("B0", np.inf))
            if reason == "omega_above_max":
                return abs(x.get("omega", np.inf))
            if reason == "time_resolution_too_low":
                return -x.get("steps_per_period", -np.inf)
            if reason == "locally_unstable":
                return max(
                    abs(x.get("stab1", np.inf)),
                    abs(x.get("stab2", np.inf)),
                )
            return 0.0
    
        examples = sorted(examples, key=sort_key)
    
        print(f"\nClosest examples for {reason}:")
        for x in examples[:n]:
            line = []
            for k in keys:
                if k in x:
                    v = x[k]
                    if isinstance(v, float):
                        line.append(f"{k}={v:.4g}")
                    else:
                        line.append(f"{k}={v}")
            print("  " + ", ".join(line))

    def print_diagnostics(self, diagnostics, top_n=10):
        from collections import Counter
    
        print("\nDIAGNOSTIC SUMMARY")
        print("=" * 70)
    
        for stage in ("geometry", "branch", "time"):
            items = diagnostics.get(stage, [])
            counts = Counter(item["reason"] for item in items)
    
            print(f"\n{stage.upper()} rejections")
            if not counts:
                print("  none")
            else:
                for reason, count in counts.most_common():
                    print(f"  {reason}: {count}")

        branch_items = diagnostics.get("branch", [])
        time_items = diagnostics.get("time", [])
        
        self.print_examples(
            branch_items,
            "B0_above_max",
            keys=["y", "r", "eta1", "eta2", "root_sign", "suppress_direction", "B0", "omega", "B0_abs", "B0_abs_max"],
        )
        
        self.print_examples(
            branch_items,
            "locally_unstable",
            keys=["y", "r", "eta1", "eta2", "root_sign", "suppress_direction", "B0", "omega", "stab1", "stab2"],
        )
        
        self.print_examples(
            time_items,
            "time_resolution_too_low",
            keys=["y", "r", "eta1", "eta2", "root_sign", "suppress_direction", "B0", "omega", "period", "steps_per_period", "recommended_n_steps"],
        )    
        scans = diagnostics.get("root_scan", [])
        if scans:
            no_root = [s for s in scans if s["n_roots"] == 0]
            print("\nROOT SCAN")
            print(f"  y values scanned: {len(scans)}")
            print(f"  y values with no roots: {len(no_root)}")
    
            finite_scans = [
                s for s in scans
                if np.isfinite(s["min_abs_diode_condition"])
            ]
    
            finite_scans.sort(key=lambda s: s["min_abs_diode_condition"])
    
            print("\nBest near misses of reduced diode condition:")
            for s in finite_scans[:top_n]:
                print(
                    "  "
                    f"y={s['y']:.6g}, "
                    f"best_r={s['best_r']:.6g}, "
                    f"value={s['best_value']:.3e}, "
                    f"min_abs={s['min_abs_diode_condition']:.3e}, "
                    f"n_roots={s['n_roots']}, "
                    f"sign_changes={s['n_sign_changes']}, "
                    f"two_contours={s['both_rashba_contours_exist']}"
                )
