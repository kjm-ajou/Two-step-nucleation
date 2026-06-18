#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# MPI
from mpi4py import MPI
MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()

# ------------------------------------------------------------
# Optional: SciencePlots for nicer figures
# ------------------------------------------------------------
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except Exception:
    pass

# ------------------------
# 0) PARAMETERS
# ------------------------
T = 220.0
SCO_LIST = [2.5, 3.0, 5.0]
s_cm = 0.5
gamma_co, gamma_cm, gamma_mo = 15.4, 2.6, 12.8

f0 = 1.0e5
g0 = 2.0e7
Q  = 0.5

C1_base = 1.6e21
M = 240

t_max  = 60e-6
N_eval = 3000
t_eval = np.linspace(0.0, t_max, N_eval)

x_eval = f0 * t_eval
max_step = 1e-7
max_dx   = f0 * max_step

# ------------------------
# 1) UTILS: triangular indexing
# ------------------------
def tri_index_maps(M):
    pair2idx = {}
    idx2pair = []
    k = 0
    for i in range(1, M):
        for n in range(1, i+1):
            pair2idx[(i, n)] = k
            idx2pair.append((i, n))
            k += 1
    return pair2idx, np.array(idx2pair, dtype=int)

PAIR2IDX, IDX2PAIR = tri_index_maps(M)
Ntri = len(IDX2PAIR)

# ------------------------
# 2) Work of formation, equilibrium C, critical sizes
# ------------------------
def w_mo(i, s_mo):
    return -s_mo * i + gamma_mo * i**(2/3)

def w_cm(n):
    return -s_cm * n + gamma_cm * n**(2/3)

def w_total_ij(i, n, s_co):
    s_mo = s_co - s_cm
    return w_mo(i, s_mo) + w_cm(n)

def make_wC(s_co):
    C1 = C1_base * np.exp(s_co)
    W = np.empty(Ntri, dtype=float)
    for k, (i, n) in enumerate(IDX2PAIR):
        W[k] = w_total_ij(i, n, s_co)
    w11 = w_total_ij(1, 1, s_co)
    C = C1 * np.exp(w11 - W)
    return W, C

def crit_sizes(s_co):
    s_mo = s_co - s_cm
    i_star   = (2*gamma_mo / (3*s_mo))**3
    n_star   = (2*gamma_cm / (3*s_cm))**3
    ico_star = (2*gamma_co / (3*s_co))**3
    return int(np.ceil(i_star)), int(np.ceil(n_star)), int(np.ceil(ico_star))

# ============================================================
# 3) Attachment frequencies: OLD vs NEW
# ============================================================
def f_in_val_old(i, s_co):
    return (1.0 - Q) * f0 * np.exp(s_co) * (i**(2/3))

def g_in_val_old(n):
    return g0 * np.exp(s_cm) * (n**(2/3))

def k_ii_val_old(i, s_co):
    return Q * f0 * np.exp(s_co) * (i**(2/3))

def build_coeffs_old(W, s_co):
    a = np.zeros(Ntri); b = np.zeros(Ntri); c = np.zeros(Ntri)
    d = np.zeros(Ntri); e = np.zeros(Ntri); h = np.zeros(Ntri)
    Wmap = {(i, n): W[PAIR2IDX[(i, n)]] for (i, n) in PAIR2IDX}
    for k, (i, n) in enumerate(IDX2PAIR):
        if n < i:
            b[k] = f_in_val_old(i, s_co) / f0
            d[k] = g_in_val_old(n) / f0
            if i >= 2:
                a[k] = (f_in_val_old(i-1, s_co) / f0) * np.exp(Wmap[(i, n)] - Wmap[(i-1, n)])
            if n >= 2:
                c[k] = (g_in_val_old(n-1) / f0) * np.exp(Wmap[(i, n)] - Wmap[(i, n-1)])
        else:
            if i >= 2:
                e[k] = (k_ii_val_old(i-1, s_co) / f0) * np.exp(Wmap[(i, i)] - Wmap[(i-1, i-1)])
            h[k] = (k_ii_val_old(i, s_co) / f0)
    return a, b, c, d, e, h

def f_plus_new(i, n, s_co):
    if not (1 <= n <= i <= M-1):
        return 0.0
    w_in   = w_total_ij(i,   n, s_co)
    w_ip1n = w_total_ij(i+1, n, s_co)
    expo   = np.exp(-(w_in - w_ip1n) / 2.0)
    return (1.0 - Q) * f0 * expo * (i ** (2.0/3.0))

def g_plus_new(i, n, s_co):
    if not (1 <= n < i <= M-1):
        return 0.0
    w_in    = w_total_ij(i, n,   s_co)
    w_inp1  = w_total_ij(i, n+1, s_co)
    expo    = np.exp(-(w_in - w_inp1) / 2.0)
    return g0 * expo * (n ** (2.0/3.0))

def k_plus_new(i, s_co):
    if not (1 <= i <= M-1):
        return 0.0
    w_ii     = w_total_ij(i,   i,   s_co)
    w_ip1ip1 = w_total_ij(i+1, i+1, s_co)
    expo     = np.exp(-(w_ii - w_ip1ip1) / 2.0)
    return Q * f0 * expo * (i ** (2.0/3.0))

def build_coeffs_new(W, s_co):
    a = np.zeros(Ntri); b = np.zeros(Ntri); c = np.zeros(Ntri)
    d = np.zeros(Ntri); e = np.zeros(Ntri); h = np.zeros(Ntri)
    Wmap = {(i, n): W[PAIR2IDX[(i, n)]] for (i, n) in PAIR2IDX}
    for k, (i, n) in enumerate(IDX2PAIR):
        if n < i:
            b[k] = f_plus_new(i, n, s_co) / f0
            d[k] = g_plus_new(i, n, s_co) / f0
            if i >= 2 and n <= i-1:
                a[k] = (f_plus_new(i-1, n, s_co) / f0) * np.exp(Wmap[(i, n)] - Wmap[(i-1, n)])
            if n >= 2:
                c[k] = (g_plus_new(i, n-1, s_co) / f0) * np.exp(Wmap[(i, n)] - Wmap[(i, n-1)])
        else:
            if i >= 2:
                e[k] = (k_plus_new(i-1, s_co) / f0) * np.exp(Wmap[(i, i)] - Wmap[(i-1, i-1)])
            h[k] = k_plus_new(i, s_co) / f0
    return a, b, c, d, e, h

# ------------------------
# 4) Sparse Jacobian pattern
# ------------------------
def jac_sparsity_pattern(M):
    rows, cols = [], []
    def add(r, c):
        rows.append(r); cols.append(c)
    for k, (i, n) in enumerate(IDX2PAIR):
        add(k, k)
        if (i-1) >= 1 and n <= (i-1):
            add(k, PAIR2IDX[(i-1, n)])
        if (i+1) <= (M-1) and n <= (i+1):
            add(k, PAIR2IDX[(i+1, n)])
        if (n-1) >= 1:
            add(k, PAIR2IDX[(i, n-1)])
        if (n+1) <= i:
            add(k, PAIR2IDX[(i, n+1)])
        if n == i:
            if (i-1) >= 1:
                add(k, PAIR2IDX[(i-1, i-1)])
            if (i+1) <= (M-1):
                add(k, PAIR2IDX[(i+1, i+1)])
    S = sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(Ntri, Ntri))
    return S

JAC_SP = jac_sparsity_pattern(M)

# ------------------------
# 5) RHS
# ------------------------
def dFdx(x, F_flat, a, b, c, d, e, h):
    F = F_flat
    dF = np.zeros_like(F)
    F[PAIR2IDX[(1, 1)]] = 1.0
    for k, (i, n) in enumerate(IDX2PAIR):
        val = 0.0
        if n < i and i >= 2:
            val += a[k] * (F[PAIR2IDX[(i-1, n)]] - F[k])
        if n < i:
            Fip = F[PAIR2IDX[(i+1, n)]] if (i+1 <= M-1) else 0.0
            val -= b[k] * (F[k] - Fip)
        if n >= 2:
            val += c[k] * (F[PAIR2IDX[(i, n-1)]] - F[k])
        if n < i:
            Finp = F[PAIR2IDX[(i, n+1)]] if (n+1 <= i) else 0.0
            val -= d[k] * (F[k] - Finp)
        if n == i:
            if i >= 2:
                val += e[k] * (F[PAIR2IDX[(i-1, i-1)]] - F[k])
            Fipp = F[PAIR2IDX[(i+1, i+1)]] if (i+1 <= M-1) else 0.0
            val -= h[k] * (F[k] - Fipp)
        dF[k] = val
    return dF

# ------------------------
# 6) Fluxes I,G,K
# ------------------------
def fluxes_IGK_old(F_flat, C, s_co):
    I_map = {}; G_map = {}; K_map = {}
    F = F_flat
    for (i, n), k in PAIR2IDX.items():
        Ckn = C[k]
        if n < i:
            fval = f_in_val_old(i, s_co)
            Fip = F[PAIR2IDX[(i+1, n)]] if (i+1 <= M-1) else 0.0
            I_map[(i, n)] = fval * Ckn * (F[k] - Fip)
            gval = g_in_val_old(n)
            Finp = F[PAIR2IDX[(i, n+1)]] if (n+1 <= i) else 0.0
            G_map[(i, n)] = gval * Ckn * (F[k] - Finp)
        if n == i:
            kval = k_ii_val_old(i, s_co)
            Fipp = F[PAIR2IDX[(i+1, i+1)]] if (i+1 <= M-1) else 0.0
            K_map[(i, i)] = kval * Ckn * (F[k] - Fipp)
    return I_map, G_map, K_map

def fluxes_IGK_new(F_flat, C, s_co):
    I_map = {}; G_map = {}; K_map = {}
    F = F_flat
    for (i, n), k in PAIR2IDX.items():
        Ckn = C[k]
        if n < i:
            fval = f_plus_new(i, n, s_co)
            Fip = F[PAIR2IDX[(i+1, n)]] if (i+1 <= M-1) else 0.0
            I_map[(i, n)] = fval * Ckn * (F[k] - Fip)
            gval = g_plus_new(i, n, s_co)
            Finp = F[PAIR2IDX[(i, n+1)]] if (n+1 <= i) else 0.0
            G_map[(i, n)] = gval * Ckn * (F[k] - Finp)
        if n == i:
            kval = k_plus_new(i, s_co)
            Fipp = F[PAIR2IDX[(i+1, i+1)]] if (i+1 <= M-1) else 0.0
            K_map[(i, i)] = kval * Ckn * (F[k] - Fipp)
    return I_map, G_map, K_map

# ------------------------
# 7) Nucleation rates
# ------------------------
def Js_from_fluxes(I_map, G_map, K_map, s_co):
    i_star, n_star, ico_star = crit_sizes(s_co)
    gi = lambda i, n: G_map.get((i, n), 0.0)
    ii = lambda i, n: I_map.get((i, n), 0.0)
    kk = lambda i:    K_map.get((i, i), 0.0)
    J_dcom = sum(ii(i_star, n) for n in range(1, i_star+1))
    for i in range(i_star+1, M):
        J_dcom += ii(i, i) - gi(i, i-1)
    J_d = ii(i_star, 1) - sum(gi(i, 1) for i in range(i_star+1, M))
    J_com = J_dcom - J_d
    J_c = kk(ico_star)
    for i in range(ico_star+1, M):
        J_c += gi(i, i-1) - ii(i, i)
    J_c_1S = kk(ico_star)
    J_d_1S = ii(i_star, 1)
    return {
        "J_d+com": J_dcom,
        "J_com":   J_com,
        "J_d":     J_d,
        "J_c":     J_c,
        "J_c,1S":  J_c_1S,
        "J_d,1S":  J_d_1S
    }

def Js_from_fluxes_figure(I_map, G_map, K_map, s_co, d=None):
    i_star, n_star, ico_star = crit_sizes(s_co)
    if d is None:
        d = min(ico_star + 1, M - 1)
    def I(i, n):
        return I_map.get((i, n), 0.0)
    def G(i, n):
        return G_map.get((i, n), 0.0)
    def K_diag(i):
        return K_map.get((i, i), 0.0)
    J_cr_det = K_diag(d) + sum(G(i, i) - I(i, i) for i in range(d, M))
    J_cr = K_diag(ico_star) + sum(G(i, i) - I(i, i) for i in range(ico_star, M))
    J_tot_det = K_diag(d) + sum(I(d, n) for n in range(1, d + 1))
    J_tot = K_diag(ico_star) + sum(I(ico_star, n) for n in range(1, ico_star + 1))
    J_comp_det = (
        sum(I(d, n) for n in range(1, d + 1)) +
        sum(I(i, i) - G(i, i) for i in range(d, M))
    )
    J_comp = (
        sum(I(ico_star, n) for n in range(1, ico_star + 1)) +
        sum(I(i, i) - G(i, i) for i in range(ico_star, M))
    )
    J_comp_i_gt_istar = (
        sum(I(i_star, n) for n in range(1, i_star + 1)) +
        sum(I(i, i) - G(i, i) for i in range(i_star, M))
    )
    J_cr_met_det = (
        sum(I(i, i) for i in range(d, M)) +
        sum(G(i, d) - G(i, i) for i in range(d, M))
    )
    J_cr_met = (
        sum(I(i, i) for i in range(n_star, M)) +
        sum(G(i, n_star) - G(i, i) for i in range(n_star, M))
    )
    J_met_det = I(d, 1) - sum(G(i, 1) for i in range(d, M))
    J_met = I(i_star, 1) - sum(G(i, 1) for i in range(i_star, M))
    return {
        "J_cr_det":      J_cr_det,
        "J_cr":          J_cr,
        "J_tot_det":     J_tot_det,
        "J_tot":         J_tot,
        "J_comp_det":    J_comp_det,
        "J_comp":        J_comp,
        "J_comp_i>i*":   J_comp_i_gt_istar,
        "J_cr_met_det":  J_cr_met_det,
        "J_cr_met":      J_cr_met,
        "J_met_det":     J_met_det,
        "J_met":         J_met,
    }

def compute_Js(F_flat, C, s_co, flux_func=fluxes_IGK_old):
    I_map, G_map, K_map = flux_func(F_flat, C, s_co)
    return Js_from_fluxes(I_map, G_map, K_map, s_co)

def compute_Js_from_figure(F_flat, C, s_co, flux_func=fluxes_IGK_old, d=None):
    I_map, G_map, K_map = flux_func(F_flat, C, s_co)
    return Js_from_fluxes_figure(I_map, G_map, K_map, s_co, d=d)

# ------------------------
# 7.5) solve_ivp 진행률 표시용 래퍼 (chunk 단위)
# ------------------------
def integrate_with_progress(fun, t_grid, y0, name="", chunk=300, **kwargs):
    total = len(t_grid) - 1
    ts = [t_grid[0]]
    ys = [y0]
    t_start = t_grid[0]
    y_start = y0
    for start_idx in range(0, total, chunk):
        end_idx = min(total, start_idx + chunk)
        t_chunk = t_grid[start_idx + 1 : end_idx + 1]
        sol = solve_ivp(fun, (t_start, t_chunk[-1]), y_start,
                        t_eval=t_chunk, **kwargs)
        if not sol.success:
            raise RuntimeError(f"[{name}] solve_ivp failed at step {end_idx}: {sol.message}")
        ts.extend(sol.t)
        ys.extend(sol.y.T)
        t_start = t_chunk[-1]
        y_start = sol.y[:, -1]
        pct = end_idx / total * 100
        print(f"[{name} progress] {pct:5.1f}% ({end_idx}/{total})", flush=True)
    Y = np.column_stack(ys)
    return np.array(ts), Y

# ------------------------
# 병렬 워커: 한 시간슬라이스 flux/J 계산
# ------------------------
def _compute_step(j, F_old_col, F_new_col, C, s_co):
    I_old, G_old, K_old = fluxes_IGK_old(F_old_col, C, s_co)
    I_new, G_new, K_new = fluxes_IGK_new(F_new_col, C, s_co)
    Js_old_orig = Js_from_fluxes(I_old, G_old, K_old, s_co)
    Js_new_orig = Js_from_fluxes(I_new, G_new, K_new, s_co)
    Js_old_fig = Js_from_fluxes_figure(I_old, G_old, K_old, s_co)
    Js_new_fig = Js_from_fluxes_figure(I_new, G_new, K_new, s_co)
    return j, Js_old_orig, Js_new_orig, Js_old_fig, Js_new_fig

# ------------------------
# 8) MAIN SESSION: old vs new kinetics
# ------------------------
def run_session(s_co):
    print(f"[R{MPI_RANK}] [Stage] s_co={s_co}: precomputing W, C, coeffs, sparsity …")
    W, C = make_wC(s_co)
    a_old, b_old, c_old, d_old, e_old, h_old = build_coeffs_old(W, s_co)
    a_new, b_new, c_new, d_new, e_new, h_new = build_coeffs_new(W, s_co)
    print(f"[R{MPI_RANK}] [Done]  s_co={s_co}: coefficients (old & new) ready")

    F0 = np.zeros(Ntri, dtype=float)
    F0[PAIR2IDX[(1, 1)]] = 1.0

    fun_old = lambda x, y: dFdx(x, y, a_old, b_old, c_old, d_old, e_old, h_old)
    fun_new = lambda x, y: dFdx(x, y, a_new, b_new, c_new, d_new, e_new, h_new)

    print(f"[R{MPI_RANK}] [Solve] s_co={s_co}: integrating OLD kinetics …")
    t_old, Y_old = integrate_with_progress(
        fun_old, x_eval, F0, name=f"s_co={s_co} OLD",
        chunk=300,
        method="BDF", jac_sparsity=JAC_SP, rtol=1e-6, atol=1e-9, max_step=max_dx
    )

    print(f"[R{MPI_RANK}] [Solve] s_co={s_co}: integrating NEW kinetics …")
    t_new, Y_new = integrate_with_progress(
        fun_new, x_eval, F0, name=f"s_co={s_co} NEW",
        chunk=300,
        method="BDF", jac_sparsity=JAC_SP, rtol=1e-6, atol=1e-9, max_step=max_dx
    )

    t_x   = t_old
    t_us  = t_x / f0 * 1e6

    old_names = ["J_d+com", "J_com", "J_d", "J_c", "J_c,1S", "J_d,1S"]
    general_names = ["J_cr", "J_tot", "J_comp", "J_comp_i>i*", "J_cr_met", "J_met"]
    det_names     = ["J_cr_det", "J_tot_det", "J_comp_det", "J_cr_met_det", "J_met_det"]

    curves_orig_old = {k: np.zeros_like(t_x) for k in old_names}
    curves_orig_new = {k: np.zeros_like(t_x) for k in old_names}
    curves_gen_old  = {k: np.zeros_like(t_x) for k in general_names}
    curves_gen_new  = {k: np.zeros_like(t_x) for k in general_names}
    curves_det_old  = {k: np.zeros_like(t_x) for k in det_names}
    curves_det_new  = {k: np.zeros_like(t_x) for k in det_names}

    # MPI 사용 시 oversubscribe 방지를 위해 기본 1로 설정
    workers = 1
    print(f"[R{MPI_RANK}] [Pool] max_workers={workers}")

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j in range(len(t_x)):
            futures.append(pool.submit(_compute_step, j, Y_old[:, j], Y_new[:, j], C, s_co))
        total = len(futures)
        for idx, fut in enumerate(as_completed(futures), 1):
            j, Js_old_orig, Js_new_orig, Js_old_fig, Js_new_fig = fut.result()
            for name in old_names:
                curves_orig_old[name][j] = Js_old_orig[name]
                curves_orig_new[name][j] = Js_new_orig[name]
            for name in general_names:
                curves_gen_old[name][j] = Js_old_fig[name]
                curves_gen_new[name][j] = Js_new_fig[name]
            for name in det_names:
                curves_det_old[name][j] = Js_old_fig[name]
                curves_det_new[name][j] = Js_new_fig[name]
            if idx % 10 == 0 or idx == total:
                pct = idx / total * 100.0
                active = sum(1 for f in futures if f.running())
                print(f"[R{MPI_RANK}] [Progress] {pct:5.1f}% ({idx}/{total}) done, active workers ~{active}/{workers}", flush=True)

    print(f"[R{MPI_RANK}] [Done]  s_co={s_co}: OLD vs NEW kinetics — all rates computed")

    t_final_us = t_x[-1] / f0 * 1e6
    print(f"\n[R{MPI_RANK}] [Rates] s_co={s_co}: final J values at t={t_final_us:.3f} μs")
    for name in general_names:
        print(f"    {name:12s} (old) = {curves_gen_old[name][-1]: .5e}")
    for name in general_names:
        print(f"    {name:12s} (new) = {curves_gen_new[name][-1]: .5e}")
    print()

    # 플롯은 랭크 0만
    if MPI_RANK == 0:
        label_map_orig = {
            "J_d+com": "J_total(old def.)",
            "J_com":   "J_cluster(old def.)",
            "J_d":     "J_M(old def.)",
            "J_c":     "J_C(old def.)",
            "J_c,1S":  "J_C,1S(old def.)",
            "J_d,1S":  "J_M,1S(old def.)"
        }

        plt.figure(figsize=(7.8, 4.6))
        for name in old_names:
            plt.plot(t_us, curves_orig_old[name], label=f"{label_map_orig[name]} (old)")
            plt.plot(t_us, curves_orig_new[name], linestyle="--", label=f"{label_map_orig[name]} (new)")
        plt.xlabel(r"$t$ ($\mu$s)")
        plt.ylabel(r"$J(t)$")
        plt.title(f"Original J-set: OLD vs NEW kinetics, s_co={s_co}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7.8, 4.6))
        for name in general_names:
            plt.plot(t_us, curves_gen_old[name], label=f"{name} (old)")
            plt.plot(t_us, curves_gen_new[name], linestyle="--", label=f"{name} (new)")
        plt.xlabel(r"$t$ ($\mu$s)")
        plt.ylabel(r"$J(t)$")
        plt.title(f"General nucleation rates: OLD vs NEW kinetics, s_co={s_co}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7.8, 4.6))
        for name in det_names:
            plt.plot(t_us, curves_det_old[name], label=f"{name} (old)")
            plt.plot(t_us, curves_det_new[name], linestyle="--", label=f"{name} (new)")
        plt.xlabel(r"$t$ ($\mu$s)")
        plt.ylabel(r"$J(t)$")
        plt.title(f"Detectable nucleation rates: OLD vs NEW kinetics, s_co={s_co}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()

# ------------------------
# 10) RUN over all s_co in SCO_LIST (MPI 분할)
# ------------------------
if __name__ == "__main__":
    if MPI_RANK == 0:
        print(f"[MPI] size={MPI_SIZE}, logical CPUs={os.cpu_count()}")
        if MPI_SIZE != 6:
            print(f"[MPI] 요청한 6코어와 다릅니다. mpiexec -n 6 ... 로 실행하세요.")
    MPI_COMM.Barrier()

    # 각 랭크가 SCO_LIST를 나눠 맡음
    my_sco_list = [sco for idx, sco in enumerate(SCO_LIST) if idx % MPI_SIZE == MPI_RANK]
    for sco in my_sco_list:
        run_session(sco)

    MPI_COMM.Barrier()
    if MPI_RANK == 0:
        print("[MPI] all ranks done.")
