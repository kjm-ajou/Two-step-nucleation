"""
Streamlit-independent logic for the two-step nucleation calculator.

Keeps all physics/orchestration here so it can be tested without a browser.
app.py is a thin UI wrapper that calls these functions.

Unit convention at the UI: dmu [eV/atom], sigma [J/m^2], D [m^2/s],
atomic volume [A^3/atom], jump length [A]. Converted to SI here.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import general_support as gs
import general_plots as gp

A3_TO_M3 = 1e-30
A_TO_M = 1e-10

DEFAULT_SCALARS = dict(
    element="Fe (FCC -> amorphous -> BCC)",
    atomic_volume_A3=11.55,
    jump_f_A=2.5738686835190326,
    jump_g_A=2.5,
    jump_k_A=2.5738686835190326,
    max_size=140,
)

# Fe example: 160 K row = validated baseline, 200 K row = illustrative placeholder.
DEFAULT_ANCHORS = pd.DataFrame({
    "T_K":       [160.0, 200.0],
    "dmu_mo_eV": [0.011249300049233798, 0.0105],
    "dmu_cm_eV": [0.07918809747686367, 0.0720],
    "sigma_mo":  [0.025261854261958074, 0.0250],
    "sigma_cm":  [0.22455353123171592, 0.2200],
    "sigma_co":  [0.24981538549367402, 0.2450],
    "D_f":       [1.3294253328618815e-15, 4.0e-15],
    "D_g":       [1.6003618534327141e-13, 5.0e-13],
    "D_k":       [1.3294253328618815e-15, 4.0e-15],
})

ANCHOR_COLS = ["T_K", "dmu_mo_eV", "dmu_cm_eV", "sigma_mo", "sigma_cm", "sigma_co", "D_f", "D_g", "D_k"]


def build_user(scalars, anchors_df):
    """Build the USER dict (SI units) from UI scalars + an anchor DataFrame."""
    df = anchors_df.copy()
    for c in ANCHOR_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ANCHOR_COLS].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    df = df.sort_values("T_K").reset_index(drop=True)

    def col(name):
        return [(float(t), float(v)) for t, v in zip(df["T_K"], df[name])]

    user = dict(
        element=str(scalars["element"]),
        atomic_volume_m3=float(scalars["atomic_volume_A3"]) * A3_TO_M3,
        jump_f_m=float(scalars["jump_f_A"]) * A_TO_M,
        jump_g_m=float(scalars["jump_g_A"]) * A_TO_M,
        jump_k_m=float(scalars["jump_k_A"]) * A_TO_M,
        max_size=int(scalars["max_size"]),
        dmu_mo_eV=col("dmu_mo_eV"), dmu_cm_eV=col("dmu_cm_eV"),
        sigma_mo=col("sigma_mo"), sigma_cm=col("sigma_cm"), sigma_co=col("sigma_co"),
        D_f=col("D_f"), D_g=col("D_g"), D_k=col("D_k"),
    )
    return user, df


def validate(user, df):
    errs = []
    if len(df) < 2:
        errs.append("Provide at least 2 temperature rows (>= 2 needed to fit the T-dependence).")
    for name in ("D_f", "D_g", "D_k"):
        if any(v <= 0 for _, v in user[name]):
            errs.append(f"{name} values must be positive (Arrhenius fit uses ln D).")
    for name in ("sigma_mo", "sigma_cm", "sigma_co"):
        if any(v <= 0 for _, v in user[name]):
            errs.append(f"{name} values must be positive.")
    if user["atomic_volume_m3"] <= 0:
        errs.append("Atomic volume must be positive.")
    if user["max_size"] < 30:
        errs.append("Grid bound max_size should be >= 30.")
    return errs


def nondim_summary(user, T_iso):
    """Dimensionless parameters + Arrhenius activation energies at T_iso (all floats)."""
    params_at, fits = gs.build_param_model(user)
    nd = gs.nondim_at(float(T_iso), fits, user["atomic_volume_m3"])
    nd = {k: float(v) for k, v in nd.items()}
    nd["Q_Df_eV"] = float(fits["D_f"].Q_eV)
    nd["Q_Dg_eV"] = float(fits["D_g"].Q_eV)
    nd["Q_Dk_eV"] = float(fits["D_k"].Q_eV)
    return nd


def grid_check(user, T_list):
    params_at, _ = gs.build_param_model(user)
    rows, worst, suggested = gs.grid_report(params_at, sorted(set(float(t) for t in T_list)), user["max_size"])
    df = pd.DataFrame(rows, columns=["T (K)", "s_co", "i*", "n*", "i_co*"])
    adequate = user["max_size"] >= 3 * worst
    return df, int(worst), int(suggested), bool(adequate)


def run_isothermal(user, T_iso):
    params_at, _ = gs.build_param_model(user)
    c1 = 1.0 / user["atomic_volume_m3"]
    I = gs.iso_run(params_at, float(T_iso), t_end=1e-2, n_time=70, atol=c1 * 1e-18)
    fig = gp.plot_iso(I, user["max_size"], element=user["element"])
    rates = {k: float(I["stat_rates"][k]) for k in ("J_d", "J_com", "J_c")}
    return rates, dict(I["crit"]), fig


def run_quench(user, T_hot, T_cold, total_time_s, n_seg):
    params_at, _ = gs.build_param_model(user)
    B = gs.noniso_run(params_at, float(T_hot), float(T_cold), total_time_s=float(total_time_s),
                      n_seg=int(n_seg), max_size=user["max_size"], sub_points=3,
                      compare_stationary=True, verbose=False)
    fig = gp.plot_noniso(B, element=user["element"])
    return {"n_fail": int(B["n_fail"]), "n_points": int(len(B["t"]))}, fig
