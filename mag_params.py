import numpy as np 
from scipy.special import jv
from scipy.interpolate import PchipInterpolator

class MagneticFormulae:
    # This class is used to construct the exchange and damping parameters 
    # used in Eqs. (82-83) of the paper.
    def __init__(self, y, theta):
        self.y = float(y)
        self.theta = float(theta)
    
    # Bessel function shorthands
    def bess_J0(self, x):
        return jv(0, x)

    def bess_J1(self, x):
        return jv(1, x)

    def k_FR(self, x, s):
        return x * (1.0 - s * self.y)
    
    # Common combinations of Bessel functions as used in the paper.
    def damp_pref(self, s1, s2):
        return (1.0 - s1 * self.y) * (1.0 - s2 * self.y)

    def exch_pref(self, s1, s2, weak_soc=False):
        if self.y >= 1:
            raise ValueError("MagneticFormulae is designed for systems with 0 <= y < 1")
        if weak_soc == True:
            return 1 
        else:
            return np.sqrt(self.damp_pref(s1, s2))


    def bessel_00(self, x, s1, s2):
        return self.bess_J0(self.k_FR(x, s1)) * self.bess_J0(self.k_FR(x, s2))

    def bessel_11(self, x, s1, s2):
        return self.bess_J1(self.k_FR(x, s1)) * self.bess_J1(self.k_FR(x, s2))

    def bessel_01(self, x, s1, s2):
        return self.bess_J1(self.k_FR(x, s1)) * self.bess_J0(self.k_FR(x, s2))
    
    # Components of the damping tensor.
    def A_xx(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a += pref * self.bessel_00(x, s1, s2)
                a -= pref * s1 * s2 * np.cos(2 * self.theta) * self.bessel_11(x, s1, s2)
        return a

    def A_yy(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a += pref * self.bessel_00(x, s1, s2)
                a += pref * s1 * s2 * np.cos(2 * self.theta) * self.bessel_11(x, s1, s2)
        return a

    def A_zz(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a += pref * self.bessel_00(x, s1, s2)
                a -= pref * s1 * s2 * self.bessel_11(x, s1, s2)
        return a

    def A_xy(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a -= 0.5 * pref * s1 * s2 * np.sin(2 * self.theta) * self.bessel_11(x, s1, s2)
        return a

    def A_xz(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a -= pref * (
                    s2 * self.bessel_01(x, s2, s1)
                    + s1 * self.bessel_01(x, s1, s2)
                ) * np.cos(self.theta)
        return a

    def A_yz(self, x):
        a = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                pref = self.damp_pref(s1, s2)
                a += pref * (
                    s2 * self.bessel_01(x, s2, s1)
                    + s1 * self.bessel_01(x, s1, s2)
                ) * np.sin(self.theta)
        return a
    
    # Construct the full tensor
    def A_tensor(self, x):
        a_xx = self.A_xx(x)
        a_yy = self.A_yy(x)
        a_zz = self.A_zz(x)
        a_xy = self.A_xy(x)
        a_xz = self.A_xz(x)
        a_yz = self.A_yz(x)

        return np.array([
            [a_xx, a_xy, a_xz],
            [a_xy, a_yy, a_yz],
            [-a_xz, -a_yz, a_zz],
        ], dtype=float) / 4.0
    
    # Decaying prefactor, common (in weak SOC limit) to all components.
    def F(self, x):
        return np.sin(2 * x) / x**2
    
    def J_xx(self, x, weak_soc=False):
        th = self.theta 
        y = self.y 
        
        j = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += self.exch_pref(s1, s2, weak_soc=weak_soc)*np.sin(x*(2 - (s1 + s2)*y))*(1 + s1*s2*np.cos(2*th))
        return j / (x**2)

    def J_yy(self, x, weak_soc=False):
        th = self.theta 
        y = self.y 

        j = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += self.exch_pref(s1, s2, weak_soc=weak_soc)*np.sin(x*(2 - (s1 + s2)*y))*(1 - s1*s2*np.cos(2*th))
        return j / (x**2)
    def J_zz(self, x, weak_soc=False):
        th = self.theta 
        y = self.y 

        j = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += self.exch_pref(s1, s2, weak_soc=weak_soc)*np.sin(x*(2 - (s1 + s2)*y))*(1 + s1*s2)

        return j / (x**2)

    def J_xz(self, x, weak_soc=False):
        th = self.theta 
        y = self.y 

        j = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += self.exch_pref(s1, s2, weak_soc=weak_soc)*np.cos(x*(2 - (s1 + s2)*y))*(s1 + s2)*np.cos(th)

        return j / (x**2)

    def J_yz(self, x, weak_soc=False):
        th = self.theta 
        y = self.y 

        j = 0.0

        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += -self.exch_pref(s1, s2, weak_soc=weak_soc)*np.cos(x*(2 - (s1 + s2)*y))*(s1 + s2)*np.sin(th)

        return j / (x**2)

    def J_xy(self, x, weak_soc=False):
        th = self.theta
        y = self.y 

        j = 0.0

        for s1 in (+1, -1):
            for s2 in (+1, -1):
                j += self.exch_pref(s1, s2, weak_soc=weak_soc)*s1*s2*np.sin(x*(2 - (s1 + s2)*y))*np.sin(2*th) / 2

        return j / (x**2)

    #def J_xx(self, x):
    #    th = self.theta
    #    y = self.y
    #    return 1.0 - np.cos(2 * th) + np.cos(2 * y * x) * (1.0 + np.cos(2 * th))

    #def J_yy(self, x):
    #    th = self.theta
    #    y = self.y
    #    return 1.0 + np.cos(2 * th) + np.cos(2 * y * x) * (1.0 - np.cos(2 * th))

    #def J_zz(self, x):
    #    return 2.0 * np.cos(2 * self.y * x)

    #def J_xy(self, x):
    #    return 0.5 * np.sin(2 * self.theta) * (np.cos(2 * self.y * x) - 1.0)

    #def J_xz(self, x):
    #    return np.cos(self.theta) * np.sin(2 * self.y * x)

    #def J_yz(self, x):
    #    return -np.sin(self.theta) * np.sin(2 * self.y * x)
    
    # This version now doesn't multiply by the range function,
    # since we only get an overall prefactor if we take (1 - s1*y)(1 - s2*y) ~ 1,
    # i.e. weak SOC limit.
    def K_tensor(self, x):
        return np.array([
            [self.J_xx(x), self.J_xy(x), self.J_xz(x)],
            [self.J_xy(x), self.J_yy(x), self.J_yz(x)],
            [-self.J_xz(x), -self.J_yz(x), self.J_zz(x)],
        ], dtype=float)

    #def K_tensor(self, x):
    #    return self.F(x) * self.K_tensor_reduced(x)
    
    # Compute coefficients excluding the overall prefactors
    # of K and alpha.
    def coeffs_reduced(self, x):
        A = self.A_tensor(x)
        return {
            "J0": self.J_xx(x),
            "J1": self.J_yy(x),
            "D": self.J_xz(x),
            "a0": A[0, 0],
            "a1": A[0, 2],
        }
    
    # Dimensionless version of the diode condition. For the full 
    # physical version, one should multiply by the overall prefactor K*\alpha
    def diode_condition(self, x):
        c = self.coeffs_reduced(x)
        return c["D"] * c["a1"] + c["J0"] * c["a0"]
    
    # A handy shortcut to measure how large each of the terms 
    # in the diode cndition is
    def diode_terms_reduced(self, x):
        c = self.coeffs_reduced(x)
        term_D = c["D"] * c["a1"]
        term_J = c["J0"] * c["a0"]
        denom = max(abs(term_D), abs(term_J), 1e-300)
        return term_D, term_J, term_D + term_J, abs(term_D + term_J) / denom

    def make_params(self, x, alpha_scale=1.0, K_scale=1.0):
        return MagneticParams(
            x=x,
            y=self.y,
            theta=self.theta,
            K12_dimless=self.K_tensor(x),
            A12_dimless=self.A_tensor(x),
            alpha_scale=alpha_scale,
            K_scale=K_scale,
        )


class MagneticParams:
    def __init__(self, x, y, theta, K12_dimless, A12_dimless, alpha_scale=1.0, K_scale=1.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.alpha_scale = float(alpha_scale)
        self.K_scale = float(K_scale)

        self.K12_dimless = np.array(K12_dimless, dtype=float)
        self.A12_dimless = np.array(A12_dimless, dtype=float)

        self.K12 = self.K_scale * self.K12_dimless
        self.K21 = self.K12.T

        self.A12 = self.alpha_scale * self.A12_dimless
        self.A21 = self.A12.T

    def ex_and_damp_coeffs(self):
        return {
            "J0": self.K12[0, 0],
            "J1": self.K12[1, 1],
            "D": self.K12[0, 2],
            "a0": self.A12[0, 0],
            "a1": self.A12[0, 2],
        }

    def ex_and_damp_coeffs_dimless(self):
        return {
            "J0": self.K12_dimless[0, 0],
            "J1": self.K12_dimless[1, 1],
            "D": self.K12_dimless[0, 2],
            "a0": self.A12_dimless[0, 0],
            "a1": self.A12_dimless[0, 2],
        }
    
    def print_mag_params(self, mode="dimfull"):
        if mode == "dimfull":
            result = self.ex_and_damp_coeffs()
            print("Exchange and damping coefficients, dimensionful: ")
            print(f"J0 = {result['J0']}")
            print(f"J1 = {result['J1']}")
            print(f"D = {result['D']}")
            print(f"a0 = {result['a0']}")
            print(f"a1 = {result['a1']}")
            print(f"Leading diode term: {np.abs(result['D'] / result['a0'])}")
            print(f"FM resonance Omega0^2 = {result['J0']**2 + result['D']**2}")
            print(f"AFM resonance Omega1^2 = {result['J1']**2 - (result['J0']**2 + result['D']**2)}")
            return
        if mode == "dimless":
            result = self.ex_and_damp_coeffs_dimless()
            print("Exchange and damping coefficients, dimensionless: ")
            print(f"J0 = {result['J0']}")
            print(f"J1 = {result['J1']}")
            print(f"D = {result['D']}")
            print(f"a0 = {result['a0']}")
            print(f"a1 = {result['a1']}")
            print(f"Leading diode term: {np.abs(result['D'] / result['a0'])}")
            print(f"FM resonance Omega0^2 = {result['J0']**2 + result['D']**2}")
            print(f"AFM resonance Omega1^2 = {result['J1']**2 - (result['J0']**2 + result['D']**2)}")
            return
        raise ValueError("print_mag_params: `mode` must be one of 'dimfull' or 'dimless'")
 
    def diode_metric(self):
        c = self.ex_and_damp_coeffs()
        return c["D"] * c["a1"] + c["J0"] * c["a0"]
    
    # Compute the different gaps which are found in the FM vs. AFM cases.
    def energy_gaps(self):
        c = self.ex_and_damp_coeffs()
        Omega0 = np.sqrt(c["D"]**2 + c["J0"]**2)
        Omega1_sq = c["J1"]**2 - c["D"]**2 - c["J0"]**2
        Omega1 = np.sqrt(Omega1_sq) if Omega1_sq >= 0.0 else np.nan
        return Omega0, Omega1, Omega1_sq
    
    def candidate_B0(self, eta1, eta2, root_sign, suppress_direction, S):
        if suppress_direction not in ("12", "21"):
            raise ValueError("suppress_direction must be '12' or '21'")
        if root_sign not in (-1, +1):
            raise ValueError("root_sign must be -1 or +1")
        if eta1 != eta2:
            raise ValueError("Antiparallel branches are disabled for the finite-SOC formulas")
    
        c = self.ex_and_damp_coeffs()
        
        D = c["D"]
        J1 = c["J1"]
        a0 = c["a0"]
        if abs(a0) < 1e-15:
            raise ValueError("a0 is too small for field tuning")
        sigma = +1 if suppress_direction == "21" else -1

        Omega0 = np.sqrt(D**2 + c["J0"]**2)       
        leading = -D*sigma / a0 
        second = S*(J1*(eta1 + eta2) + (1 + eta1*eta2)*root_sign*Omega0) / 2

        return leading - second
   
    def branch_resonance(self, B0, eta1, eta2, root_sign, S):
        if root_sign not in (-1, +1):
            raise ValueError("root_sign must be -1 or +1")
        if eta1 != eta2:
            raise ValueError("Antiparallel branches are disabled for the finite-SOC formulas")
    
        c = self.ex_and_damp_coeffs()
        Omega0 = np.sqrt(c["D"]**2 + c["J0"]**2)
        J1 = c["J1"]
    
        return (
            -B0
            - 0.5 * S * J1 * (eta1 + eta2)
            - root_sign * S * Omega0
        )
    
    def branch_energy(self, B0, S, eta1, eta2):
        c = self.ex_and_damp_coeffs()
        return -c["J1"] * S**2 * eta1 * eta2 - B0 * S * (eta1 + eta2)

    def branch_stability(self, B0, S, eta1, eta2):
        c = self.ex_and_damp_coeffs()
        H1 = B0 + c["J1"] * S * eta2
        H2 = B0 + c["J1"] * S * eta1
        return eta1 * H1, eta2 * H2

    def classify_branch(self, B0, S, eta1, eta2, tol=1e-12):
        branches = [(+1, +1), (-1, -1)]
        energies = {
            branch: self.branch_energy(B0, S, branch[0], branch[1])
            for branch in branches
        }
        # E is the ground state if it has a lower energy than the other
        # three branches. It must also be (locally) dynamically stable.
        E = energies[(eta1, eta2)]
        Emin = min(energies.values())
        stab1, stab2 = self.branch_stability(B0, S, eta1, eta2)
        return {
            "energy": E,
            "energies": energies,
            "is_ground_state": E <= Emin + tol,
            "stab1": stab1,
            "stab2": stab2,
            "is_locally_stable": stab1 > tol and stab2 > tol,
        }
    
    # Construct the 6x6 damping covariance matrix;
    # if this matrix is positive semi-definite, it 
    # is in the right form to define a stochastic process according to 
    # <xi_i xi_j> ~ \alpha_{ij}
    def alpha6(self, alpha_loc):
        A11 = alpha_loc * np.eye(3)
        A22 = alpha_loc * np.eye(3)
        return np.block([
            [A11, self.A12],
            [self.A21, A22],
        ])
    
    # Perform the covariance check.
    def damping_psd_info(self, alpha_loc):
        A6 = self.alpha6(alpha_loc)
        A6s = 0.5 * (A6 + A6.T)
        eig = np.linalg.eigvalsh(A6s)
        return eig[0], eig

    def effective_fields(self, S1, S2, B_ext1, B_ext2):
        B1 = B_ext1 + self.K12 @ S2
        B2 = B_ext2 + self.K21 @ S1
        return B1, B2


    def weak_damping_chi_at_resonance(self, B0, eta1, eta2, root_sign, S):
        omega0 = self.branch_resonance(
            B0=B0,
            eta1=eta1,
            eta2=eta2,
            root_sign=root_sign,
            S=S,
        )

        c = self.ex_and_damp_coeffs()
        D = c["D"]
        J0 = c["J0"]
        a0 = c["a0"]
        a1 = c["a1"]

        chi12 = eta1*S*(-D + 1j*J0 + a0*omega0 + 1j*a1*omega0)
        chi21 = eta2*S*(D + 1j*J0 + a0*omega0 - 1j*a1*omega0)

        return chi12, chi21, omega0

    def optimal_resonance_frequency_for_suppression(self, eta1, suppress_direction):
        c = self.ex_and_damp_coeffs()
        D = c["D"]
        J0 = c["J0"]
        a0 = c["a0"]
        a1 = c["a1"]

        denom = a0**2 + a1**2
        if denom <= 0.0:
            raise ValueError("non-local damping coefficients are too small to optimise suppression")
        
        target = (a0 * D - a1 * J0) / denom

        if suppress_direction == "21":
            return target
        if suppress_direction == "12":
            return -target
        raise ValueError("suppress_direction must be '12' or '21'")

    def optimal_B0_for_suppression(self, eta1, eta2, root_sign, suppress_direction, S):
        omega_target = self.optimal_resonance_frequency_for_suppression(
            eta1=eta1,
            suppress_direction=suppress_direction,
        )
        omega_at_zero_field = self.branch_resonance(
            B0=0.0,
            eta1=eta1,
            eta2=eta2,
            root_sign=root_sign,
            S=S,
        )
        return omega_at_zero_field - omega_target 

    def weak_damping_contrast_at_optimal_field(self, eta1, eta2, root_sign, suppress_direction, S):
        B0 = self.optimal_B0_for_suppression(
            eta1=eta1,
            eta2=eta2,
            root_sign=root_sign,
            suppress_direction=suppress_direction,
            S=S,
        )
        chi12, chi21, omega0 = self.weak_damping_chi_at_resonance(
            B0=B0,
            eta1=eta1,
            eta2=eta2,
            root_sign=root_sign,
            S=S,
        )

        abs_chi12 = abs(chi12)
        abs_chi21 = abs(chi21)
        eps = 1e-300

        if suppress_direction == "21":
            suppressed_abs = abs_chi12
            allowed_abs = abs_chi21
        elif suppress_direction == "12":
            suppressed_abs = abs_chi21
            allowed_abs = abs_chi12
        else:
            raise ValueError("suppress_direction must be '12' or '21'")

        isolation = allowed_abs / max(suppressed_abs, eps)

        return {
            "B0": B0,
            "omega": omega0,
            "chi12": chi12,
            "chi21": chi21,
            "abs_chi12": abs_chi12,
            "abs_chi21": abs_chi21,
            "suppressed_abs": suppressed_abs,
            "allowed_abs": allowed_abs,
            "isolation": isolation,
            "isolation_dB": 20.0 * np.log10(max(isolation, eps)),
            "chi12_over_chi21": abs_chi12 / max(abs_chi21, eps),
            "chi21_over_chi12": abs_chi21 / max(abs_chi12, eps),
        }


    def print_params(self):
        print("=" * 70)
        print(f"Magnetic parameters for x={self.x}; y={self.y}; theta={self.theta}")
        print("=" * 70)
        print("K12:")
        print(self.K12)
        print("A12:")
        print(self.A12)
        c = self.ex_and_damp_coeffs()
        Omega0, Omega1, Omega1_sq = self.energy_gaps()
        print("Paper-style coefficients, if this geometry is appropriate:")
        print("J0:", c["J0"])
        print("J1:", c["J1"])
        print("D:", c["D"])
        print("a0:", c["a0"])
        print("a1:", c["a1"])
        print("Omega0:", Omega0)
        print("Omega1:", Omega1)
        print("Omega1_sq:", Omega1_sq)
        print("diode metric:", self.paper_diode_metric())
        print("=" * 70)


class NumericalExchangeTable(MagneticFormulae):
    def __init__(self, y, theta, xs, values):
        super().__init__(y=y, theta=theta)
        self.Jxx_interp = PchipInterpolator(xs, values["xx"])
        self.Jyy_interp = PchipInterpolator(xs, values["yy"])
        self.Jzz_interp = PchipInterpolator(xs, values["zz"])
        self.Jxy_interp = PchipInterpolator(xs, values["xy"])
        self.Jxz_interp = PchipInterpolator(xs, values["xz"])
        self.Jyz_interp = PchipInterpolator(xs, values["yz"])


    def J_xx(self, x):
        return self.Jxx_interp(x)
    def J_yy(self, x):
        return self.Jyy_interp(x)
    def J_zz(self, x):
        return self.Jzz_interp(x)

    def J_xy(self, x):
        return self.Jxy_interp(x)
    def J_yx(self, x):
        return self.Jxy_interp(x)
    
    def J_xz(self, x):
        return self.Jxz_interp(x)
    def J_zx(self, x):
        return -self.Jxz_interp(x)

    def J_yz(self, x):
        return self.Jyz_interp(x)
    def J_zy(self, x):
        return -self.Jyz_interp(x)

    def K_tensor(self, x):
        return np.array([
            [self.J_xx(x), self.J_xy(x), self.J_xz(x)],
            [self.J_yx(x), self.J_yy(x), self.J_yz(x)],
            [self.J_zx(x), self.J_zy(x), self.J_zz(x)]
        ])
