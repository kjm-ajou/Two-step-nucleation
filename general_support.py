"""
General unary-metal two-step nucleation: support layer.

The user supplies PHYSICAL parameters (chemical-potential differences, interface
energies, diffusion coefficients, atomic volume, jump lengths) as a function of
temperature (anchor points). This module:
  - fits the T-dependence (Arrhenius for D, low-order polynomial for dmu and sigma),
  - nondimensionalizes to the Kashchiev variables (s_co, s_cm, gamma_*),
  - drives the validated Turnbull-Fisher two-step engine (model_core) for both
    isothermal and non-isothermal (segment-stepping) runs.

Nondimensionalization (per monomer, in units of kT):
    s  = dmu / kT                         (dimensionless supersaturation)
    gamma = sigma * A1 / kT,  A1 = (36*pi)^(1/3) * v^(2/3)   (monomer surface area)
Monomer number density: c1_base = 1 / v.
"""
import math
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import model_core as mc

kB_eV = 8.617333262e-5      # eV / K
eV_J = 1.602176634e-19      # J / eV


# ---------------------------------------------------------------------------
# 1. Fit the T-dependence of each physical input
# ---------------------------------------------------------------------------
def _poly_fit(points, max_deg=2):
    """Polynomial fit value(T). points = [(T, value), ...]. Degree = min(npts-1, max_deg)."""
    pts = sorted(points)
    T = np.array([p[0] for p in pts], float)
    V = np.array([p[1] for p in pts], float)
    deg = min(len(pts) - 1, max_deg)
    coeff = np.polyfit(T, V, deg)
    return lambda Tq: float(np.polyval(coeff, Tq))


def _arrhenius_fit(points):
    """Arrhenius fit D(T) = D0 exp(-Q/kB T) via linear ln D vs 1/T. points = [(T, D), ...]."""
    pts = sorted(points)
    T = np.array([p[0] for p in pts], float)
    D = np.array([p[1] for p in pts], float)
    if np.any(D <= 0):
        raise ValueError("diffusion coefficients must be positive for Arrhenius fit")
    coeff = np.polyfit(1.0 / T, np.log(D), 1)  # slope = -Q/kB, intercept = ln D0
    Q_eV = -coeff[0] * kB_eV
    D0 = math.exp(coeff[1])
    fn = lambda Tq: float(math.exp(np.polyval(coeff, 1.0 / Tq)))
    fn.Q_eV = Q_eV
    fn.D0 = D0
    return fn


def make_fits(user):
    """Build interpolating functions for every physical quantity from user anchors."""
    return {
        "dmu_mo": _poly_fit(user["dmu_mo_eV"]),
        "dmu_cm": _poly_fit(user["dmu_cm_eV"]),
        "sigma_mo": _poly_fit(user["sigma_mo"]),
        "sigma_cm": _poly_fit(user["sigma_cm"]),
        "sigma_co": _poly_fit(user["sigma_co"]),
        "D_f": _arrhenius_fit(user["D_f"]),
        "D_g": _arrhenius_fit(user["D_g"]),
        "D_k": _arrhenius_fit(user["D_k"]),
    }


def nondim_at(T, fits, v_m3):
    """Physical -> dimensionless Kashchiev parameters at temperature T."""
    kT = kB_eV * T
    s_mo = fits["dmu_mo"](T) / kT
    s_cm = fits["dmu_cm"](T) / kT
    s_co = s_mo + s_cm
    A1 = (36.0 * math.pi) ** (1.0 / 3.0) * v_m3 ** (2.0 / 3.0)  # monomer surface area (m^2)
    kT_J = kT * eV_J
    g = lambda name: fits[name](T) * A1 / kT_J
    return dict(s_co=s_co, s_cm=s_cm, s_mo=s_mo,
                gamma_mo=g("sigma_mo"), gamma_cm=g("sigma_cm"), gamma_co=g("sigma_co"))


# ---------------------------------------------------------------------------
# 2. Parameter-model factory: T -> (Paper2022Params, s_co, closure_set)
# ---------------------------------------------------------------------------
def build_param_model(user):
    fits = make_fits(user)
    v = float(user["atomic_volume_m3"])
    max_size = int(user["max_size"])
    c1_base = 1.0 / v

    def params_at(T):
        nd = nondim_at(T, fits, v)
        params = mc.Paper2022Params(
            temperature_K=float(T), s_cm=nd["s_cm"],
            gamma_mo=nd["gamma_mo"], gamma_cm=nd["gamma_cm"], gamma_co=nd["gamma_co"],
            c1_base=c1_base, q_split=0.5, max_size=max_size,
        )
        closure_set = mc.make_diffusion_turnbull_fisher_closure_set(
            f_diffusion_m2_s=fits["D_f"](T), f_jump_length_m=float(user["jump_f_m"]),
            g_diffusion_m2_s=fits["D_g"](T), g_jump_length_m=float(user["jump_g_m"]),
            k_diffusion_m2_s=fits["D_k"](T), k_jump_length_m=float(user["jump_k_m"]),
        )
        return params, nd["s_co"], closure_set

    return params_at, fits


def grid_report(params_at, T_list, max_size):
    """Critical sizes across a temperature list; warn if close to the grid bound."""
    rows = []
    worst = 0
    for T in T_list:
        params, s_co, _ = params_at(T)
        crit = mc.critical_sizes(s_co, params)
        worst = max(worst, crit["i_star"], crit["n_star"], crit["i_co_star"])
        rows.append((T, s_co, crit["i_star"], crit["n_star"], crit["i_co_star"]))
    suggested = max(60, int(math.ceil(5.0 * worst)))
    return rows, worst, suggested


# ---------------------------------------------------------------------------
# 3. Isothermal run (rates, cluster population distribution, work surface)
# ---------------------------------------------------------------------------
def iso_run(params_at, T_iso, t_end=1e-2, n_time=80, t_start=1e-12, atol=1e-30):
    params, s_co, cs = params_at(T_iso)
    time_s = np.concatenate(([0.0], np.geomspace(t_start, t_end, n_time)))

    stat = mc.solve_dimensionless_system(s_co=s_co, params=params, time_s=np.array([0.0]),
                                         method="stationary", closure_set=cs)
    stat_rates = mc.final_rates_from_solve_output(stat, s_co, params)
    crit = mc.critical_sizes(s_co, params)

    tr = mc.solve_dimensionless_system(s_co=s_co, params=params, time_s=time_s,
                                       method="BDF_linear_c", closure_set=cs,
                                       max_step_s=None, atol=atol)
    rates_t = mc.rate_time_series_from_solution_stable(tr, s_co, params)
    dens_t = mc.number_density_time_series(time_s, rates_t)

    # cluster population fields
    y = np.asarray(tr["solution"].y, float)           # f = c/c_eq, shape (Nstate, Ntime)
    w = np.asarray(tr["w"], float)
    c1 = float(tr["c1"])
    pairs = np.asarray(tr["state_pairs"], int)
    log10c_final = mc.log10_actual_concentration(y[:, -1], w, c1, s_co, params)

    # weighted centroid trajectory <i>,<n> over post-critical population
    ii = pairs[:, 0].astype(float)
    nn = pairs[:, 1].astype(float)
    cen_i, cen_n = [], []
    for j in range(y.shape[1]):
        c = mc.restore_actual_concentration(y[:, j], w, c1, s_co, params)
        mask = ii >= max(2, crit["i_star"])  # post-critical-ish
        wsum = np.sum(c[mask])
        if wsum > 0:
            cen_i.append(np.sum(c[mask] * ii[mask]) / wsum)
            cen_n.append(np.sum(c[mask] * nn[mask]) / wsum)
        else:
            cen_i.append(np.nan); cen_n.append(np.nan)

    return dict(params=params, s_co=s_co, crit=crit, time_s=time_s,
                stat_rates=stat_rates, rates_t=rates_t, dens_t=dens_t,
                pairs=pairs, w=w, c1=c1, log10c_final=log10c_final,
                y=y, cen_i=np.array(cen_i), cen_n=np.array(cen_n))


# ---------------------------------------------------------------------------
# 4. Non-isothermal run (segment-stepping; carry actual concentration forward)
# ---------------------------------------------------------------------------
def _build_operator(params_at, T, PAIR, PAIRS, TOPO, NST):
    params, s_co, cs = params_at(T)
    w, c_eq, c1 = mc.make_w_and_c_arrays(s_co, params, PAIRS)
    closures = mc._resolve_closure_set(params, cs)
    coeffs = mc.build_coefficients(w, s_co, params, PAIR, PAIRS, closure_set=closures)
    system = mc._build_transient_linear_system(coeffs, TOPO, PAIR, PAIRS)
    A_f = system["matrix"].tocsr()
    s_f = np.asarray(system["source"], float)
    unk = system["unknown_indices"]
    anchor = int(system["anchor_idx"])
    c_eq_u = c_eq[unk]
    A_coo = A_f.tocoo()
    lce = np.log(np.maximum(c_eq_u, 1e-300))
    lr = np.clip(lce[A_coo.row] - lce[A_coo.col], -700.0, 700.0)
    M = sp.csr_matrix((A_coo.data * np.exp(lr), (A_coo.row, A_coo.col)), shape=A_f.shape)
    s_c = c_eq_u * s_f
    return dict(params=params, s_co=s_co, closures=closures, w=w, c1=c1,
                M=M, s_c=s_c, unk=unk, anchor=anchor, c_eq_u=c_eq_u,
                rate_ref=float(cs.reference_rate_s))


def _rates_from_c(c_unknown, op, PAIR, NST):
    f_unknown = c_unknown / op["c_eq_u"]
    f_full = mc._reconstruct_full_state_trajectory(NST, op["anchor"], op["unk"], f_unknown[:, None])[:, 0]
    return mc.rate_dict_from_state_stable(f_full, op["w"], op["c1"], op["s_co"],
                                          op["params"], PAIR, closure_set=op["closures"])


def noniso_run(params_at, T_hot, T_cold, total_time_s, n_seg, max_size,
               sub_points=3, atol=None, rtol=1e-6, compare_stationary=False, verbose=True):
    PAIR, PAIRS = mc.triangular_index_maps(max_size)
    TOPO = mc.build_state_topology(PAIR, PAIRS, max_size)
    NST = len(PAIRS)
    PAIRS_arr = np.asarray(PAIRS, dtype=int)
    if atol is None:
        atol = float(params_at(T_hot)[0].c1_base) * 1e-14  # ~14 orders below monomer density

    t_edges = np.linspace(0.0, total_time_s, n_seg + 1)
    T_edges = np.linspace(T_hot, T_cold, n_seg + 1)
    op0 = _build_operator(params_at, T_edges[0], PAIR, PAIRS, TOPO, NST)
    c_unknown = np.zeros(len(op0["unk"]), float)
    unk_ref, anchor_ref = op0["unk"], op0["anchor"]

    rec = {k: [] for k in ("t", "T", "J_d", "J_com", "J_c", "N_tot", "N_i", "N_n", "n_star")}
    if compare_stationary:
        rec["J_d_stat"] = []; rec["J_com_stat"] = []
        stat_cache = {}
    n_fail = 0
    for k in range(n_seg):
        T_seg = 0.5 * (T_edges[k] + T_edges[k + 1])
        op = _build_operator(params_at, T_seg, PAIR, PAIRS, TOPO, NST)
        assert op["anchor"] == anchor_ref and np.array_equal(op["unk"], unk_ref)
        dt = t_edges[k + 1] - t_edges[k]
        r, M, s_c = op["rate_ref"], op["M"], op["s_c"]
        teval = np.linspace(0.0, dt, sub_points + 1)[1:]
        sol = solve_ivp(lambda t, c: r * (M.dot(c) + s_c), (0.0, dt), c_unknown,
                        method="BDF", jac=(r * M), t_eval=teval, rtol=rtol, atol=atol)
        if not sol.success:
            n_fail += 1
            if verbose:
                print(f"  [seg {k}] T={T_seg:.2f}K FAILED: {sol.message}")
            break
        if compare_stationary and T_seg not in stat_cache:
            p, sco, cc = params_at(T_seg)
            so = mc.solve_dimensionless_system(s_co=sco, params=p, time_s=np.array([0.0]),
                                               method="stationary", closure_set=cc)
            rr = mc.final_rates_from_solve_output(so, sco, p)
            stat_cache[T_seg] = (rr["J_d"], rr["J_com"])
        crit_seg = mc.critical_sizes(op["s_co"], op["params"])
        for j in range(sol.y.shape[1]):
            cj = sol.y[:, j]
            rd = _rates_from_c(cj, op, PAIR, NST)
            # full actual concentration (monomer pinned at c1) -> cluster-size marginals
            c_full = np.zeros(NST)
            c_full[op["unk"]] = cj
            c_full[op["anchor"]] = op["c1"]
            Ni = np.zeros(max_size + 2); np.add.at(Ni, PAIRS_arr[:, 0], c_full)
            Nn = np.zeros(max_size + 2); np.add.at(Nn, PAIRS_arr[:, 1], c_full)
            rec["t"].append(t_edges[k] + teval[j]); rec["T"].append(T_seg)
            rec["J_d"].append(rd["J_d"]); rec["J_com"].append(rd["J_com"]); rec["J_c"].append(rd["J_c"])
            rec["N_tot"].append(float(np.sum(cj)))
            rec["N_i"].append(Ni); rec["N_n"].append(Nn); rec["n_star"].append(int(crit_seg["n_star"]))
            if compare_stationary:
                rec["J_d_stat"].append(stat_cache[T_seg][0]); rec["J_com_stat"].append(stat_cache[T_seg][1])
        c_unknown = sol.y[:, -1]
        if verbose and (k % max(1, n_seg // 8) == 0 or k == n_seg - 1):
            print(f"  [seg {k:3d}] T={T_seg:7.2f}K  J_d={rec['J_d'][-1]:.3e}  "
                  f"J_com={rec['J_com'][-1]:.3e}  N_tot={rec['N_tot'][-1]:.3e}")
    out = {k: np.array(v) for k, v in rec.items()}
    out["n_fail"] = n_fail
    out["c_final"] = c_unknown
    return out
