import numpy as np

from numpy.polynomial.legendre import leggauss
from scipy.special import jv


BRANCHES = (+1, -1)
COMPONENTS = (
    ("xx", "xy", "xz"),
    ("yx", "yy", "yz"),
    ("zx", "zy", "zz"),
)


class TransformQuadratureRule:
    def __init__(
        self,
        y,
        e_max=500.0,
        n_e=900,
        n_h=160,
        n_rho=700,
        rho_scale=1.0,
    ):
        if not (0 <= y < 1):
            raise ValueError("This parametrisation assumes 0 <= y = q_R/k_F < 1.")
        if min(e_max, rho_scale) <= 0:
            raise ValueError("Use e_max > 0 and rho_scale > 0.")
        if min(n_e, n_h, n_rho) < 8:
            raise ValueError("Use at least 8 quadrature nodes in each direction.")

        self.y = float(y)
        self.e_max = float(e_max)
        self.n_e = int(n_e)
        self.n_h = int(n_h)
        self.n_rho = int(n_rho)
        self.rho_scale = float(rho_scale)

        z_rho, w_rho = leggauss(self.n_rho)
        t = 0.5 * (z_rho + 1.0)
        wt = 0.5 * w_rho
        one_minus_t = 1.0 - t
        self.rho = self.rho_scale * t / one_minus_t
        self.wrho = wt * self.rho_scale / one_minus_t**2

        z_e, w_e = leggauss(self.n_e)
        self.e = 0.5 * self.e_max * (z_e + 1.0)
        self.we = 0.5 * self.e_max * w_e

        self.h_nodes = {}
        self.h_weights = {}
        for s in BRANCHES:
            d = hole_depth_dimless(s, self.y)
            z_h, w_h = leggauss(self.n_h)
            self.h_nodes[s] = 0.5 * d * (z_h + 1.0)
            self.h_weights[s] = 0.5 * d * w_h

        self.exp_e = np.exp(-self.rho[:, None] * self.e[None, :])
        self.exp_h = {
            s: np.exp(-self.rho[:, None] * self.h_nodes[s][None, :])
            for s in BRANCHES
        }


def hole_depth_dimless(s, y):
    if s == +1:
        return 0.5 * (1.0 - y**2)
    if s == -1:
        return 0.5
    raise ValueError("s must be +1 or -1")


def B_lin(nu, s, e, x, y):
    prefactor = 1.0 - s * y / (1.0 + e)
    argument = x * (1.0 - s * y + e)
    return prefactor * jv(nu, argument)


def coeff_terms(component, sp, s, theta):
    ss = s * sp
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    c1 = np.cos(theta)
    s1 = np.sin(theta)

    if component == "xx":
        return ((0, 0, 1.0), (1, 1, -ss * c2))
    if component == "yy":
        return ((0, 0, 1.0), (1, 1, +ss * c2))
    if component == "zz":
        return ((0, 0, 1.0), (1, 1, -ss))
    if component in ("xy", "yx"):
        return ((1, 1, -0.5 * ss * s2),)
    if component == "xz":
        return ((1, 0, -sp * c1), (0, 1, -s * c1))
    if component == "zx":
        return ((1, 0, +sp * c1), (0, 1, +s * c1))
    if component == "yz":
        return ((1, 0, +sp * s1), (0, 1, +s * s1))
    if component == "zy":
        return ((1, 0, -sp * s1), (0, 1, -s * s1))

    valid = ", ".join(name for row in COMPONENTS for name in row)
    raise ValueError(f"component must be one of {valid}")


def _transforms_for_x(x, rule):
    y = rule.y
    P = {}
    H = {}

    for nu in (0, 1):
        for s in BRANCHES:
            b_e = B_lin(nu, s, rule.e, x, y)
            P[(nu, s)] = rule.exp_e @ (rule.we * b_e)

            h = rule.h_nodes[s]
            b_h = B_lin(nu, s, -h, x, y)
            H[(nu, s)] = rule.exp_h[s] @ (rule.h_weights[s] * b_h)

    return P, H


def _component_from_transforms(component, P, H, wrho, theta):
    integrand = np.zeros_like(wrho)

    for sp in BRANCHES:
        for s in BRANCHES:
            for nu_p, nu_h, c in coeff_terms(component, sp, s, theta):
                sector_a = P[(nu_p, sp)] * H[(nu_h, s)]
                sector_b = H[(nu_p, sp)] * P[(nu_h, s)]
                integrand += c * (sector_a + sector_b)

    return 2.0 * np.dot(wrho, integrand)


def K_component_dimless(component, x, y, theta=0.0, rule=None):
    rule = rule or TransformQuadratureRule(y)
    if abs(rule.y - y) > 1e-15:
        raise ValueError("The supplied quadrature rule was built for a different y.")

    P, H = _transforms_for_x(float(x), rule)
    return _component_from_transforms(component, P, H, rule.wrho, theta)


def K_tensor_dimless(x, y, theta=0.0, rule=None):
    rule = rule or TransformQuadratureRule(y)
    if abs(rule.y - y) > 1e-15:
        raise ValueError("The supplied quadrature rule was built for a different y.")

    P, H = _transforms_for_x(float(x), rule)
    out = np.zeros((3, 3), dtype=float)

    for i, row in enumerate(COMPONENTS):
        for j, component in enumerate(row):
            out[i, j] = _component_from_transforms(component, P, H, rule.wrho, theta)

    return out


def scan_components(
    components,
    xs,
    y,
    theta=0.0,
    e_max=500.0,
    n_e=900,
    n_h=160,
    n_rho=700,
    rho_scale=1.0,
):
    rule = TransformQuadratureRule(
        y,
        e_max=e_max,
        n_e=n_e,
        n_h=n_h,
        n_rho=n_rho,
        rho_scale=rho_scale,
    )
    xs = np.asarray(xs, dtype=float)
    values = {component: np.empty_like(xs) for component in components}

    for n, x in enumerate(xs, start=1):
        #print(f"{n}/{len(xs)}, x = {x:.4f}")
        P, H = _transforms_for_x(float(x), rule)

        for component in components:
            values[component][n - 1] = _component_from_transforms(
                component, P, H, rule.wrho, theta
            )

    return values
