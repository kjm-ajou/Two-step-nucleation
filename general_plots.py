"""Plotting helpers for the unary-metal two-step nucleation app/notebook.

Isothermal output (cluster-population focused):
  plot_rates_and_densities  - (a) nucleation rates, (b) cumulative number density
  plot_work_surface         - nucleation work surface contour (with i*, n*, i_co*, n=i)
  plot_population_histograms - overlaid metastable-size N_i and crystal-size N_n at a time
  plot_crystal_marginal_timeseries - crystal-size marginal vs n at several times (time colorbar)
Non-isothermal output:
  plot_noniso               - quench J(t)/J(T) with instantaneous-stationary comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, Normalize
import model_core as mc  # in the notebook this is rebound to sys.modules['__main__']


def _marg_i(c, pairs, M):
    m = np.zeros(M + 2); np.add.at(m, pairs[:, 0], c); return m  # N_i = sum_n c(i,n)


def _marg_n(c, pairs, M):
    m = np.zeros(M + 2); np.add.at(m, pairs[:, 1], c); return m  # N_n = sum_i c(i,n)


def plot_rates_and_densities(I, element=""):
    t = I["time_s"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for nm, col in [("J_d", "#1f77b4"), ("J_com", "#d62728"), ("J_c", "#2ca02c")]:
        a1.loglog(t[1:], np.maximum(I["rates_t"][nm][1:], 1e-300), color=col, lw=1.8, label=nm)
    a1.set_xlabel("time (s)"); a1.set_ylabel(r"$J$ (m$^{-3}$s$^{-1}$)")
    a1.set_title("(a) nucleation rates"); a1.legend(fontsize=9); a1.grid(alpha=0.3, which="both")
    for nm, col in [("N_d", "#1f77b4"), ("N_com", "#d62728"), ("N_c", "#2ca02c")]:
        a2.loglog(t[1:], np.maximum(I["dens_t"][nm][1:], 1e-300), color=col, lw=1.8, label=nm)
    a2.set_xlabel("time (s)"); a2.set_ylabel(r"$N$ (m$^{-3}$)")
    a2.set_title("(b) cumulative number density"); a2.legend(fontsize=9); a2.grid(alpha=0.3, which="both")
    fig.suptitle(f"{element}  (T = {I['params'].temperature_K:.0f} K)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_work_surface(I, element="", i_max=None, n_max=80):
    p = I["params"]; s_cm = p.s_cm; g_mo = p.gamma_mo; g_cm = p.gamma_cm
    s_mo = I["s_co"] - s_cm; crit = I["crit"]
    i_max = i_max or p.max_size
    iv = np.linspace(1, i_max, 250); nv = np.linspace(1, n_max, 250)
    II, NN = np.meshgrid(iv, nv)
    W = -s_mo * II + g_mo * II ** (2 / 3) - s_cm * NN + g_cm * NN ** (2 / 3)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(iv.min(), iv.max()); ax.set_ylim(nv.min(), nv.max())
    cf = ax.contourf(iv, nv, W, levels=120, alpha=0.80, cmap="viridis", zorder=1)
    lvl = np.linspace(W.min(), W.max(), 14)
    ax.contour(iv, nv, W, levels=lvl, linewidths=3.0, colors="k", alpha=0.45, zorder=25)
    cs = ax.contour(iv, nv, W, levels=lvl, linewidths=1.6, cmap="viridis", zorder=40)
    labs = ax.clabel(cs, cs.levels, inline=True, fontsize=9, fmt=lambda v: f"{v:.0f}",
                     inline_spacing=4, colors="k")
    for tl in labs:
        tl.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)])
    d0 = max(iv.min(), nv.min()); d1 = min(iv.max(), nv.max())
    xd = np.linspace(d0, d1, 400)
    ax.plot(xd, xd, lw=2.4, color="black", label="$n = i$", zorder=1500)
    ax.scatter([crit["i_star"]], [crit["n_star"]], marker="*", s=170, color="#1f77b4",
               edgecolor="k", lw=0.6, zorder=3000, label="$i^{*},\\,n^{*}$")
    ax.scatter([crit["i_co_star"]], [crit["i_co_star"]], marker="s", s=90, color="#ff7f0e",
               edgecolor="k", lw=0.6, zorder=3000, label="$i_{co}^{*}$")
    ax.axvline(crit["i_star"], ls="--", lw=1.8, color="black", zorder=1500)
    ax.axhline(crit["n_star"], ls="--", lw=1.8, color="black", zorder=1500)
    ax.set_xlabel("i  (metastable cluster size)", fontsize=13, fontweight="bold")
    ax.set_ylabel("n  (crystal cluster size)", fontsize=13, fontweight="bold")
    ax.set_title(f"Nucleation work surface — {element}", fontsize=14, fontweight="bold")
    leg = ax.legend(frameon=True, facecolor="white", edgecolor="0.8", framealpha=1.0, fontsize=11)
    leg.set_zorder(4000)
    ax.grid(True, ls="--", alpha=0.4)
    fig.colorbar(cf, ax=ax, label=r"nucleation work  $w_{i,n}/k_BT$")
    fig.tight_layout()
    return fig


def plot_population_histograms(I, t_target=5e-6, element="", size_max=40):
    t = I["time_s"]; j = int(np.argmin(np.abs(t - t_target)))
    c = mc.restore_actual_concentration(I["y"][:, j], I["w"], I["c1"], I["s_co"], I["params"])
    M = I["params"].max_size
    Ni = _marg_i(c, I["pairs"], M); Nn = _marg_n(c, I["pairs"], M)
    s = np.arange(1, size_max + 1)
    lNi = np.log10(np.maximum(Ni[1:size_max + 1], 1e-300))
    lNn = np.log10(np.maximum(Nn[1:size_max + 1], 1e-300))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(s, lNi, width=1.0, align="center", color="#9ecae1", alpha=0.65, edgecolor="#3182bd",
           lw=0.4, label=r"metastable-size population $N_i=\sum_n c(i,n)$")
    ax.bar(s, lNn, width=1.0, align="center", color="#fcbba1", alpha=0.65, edgecolor="#de2d26",
           lw=0.4, label=r"crystal-size population $N_n=\sum_i c(i,n)$")
    tt = t[j]; unit = f"{tt*1e6:.1f} \u03bcs" if tt < 1e-3 else f"{tt*1e3:.1f} ms"
    ax.set_xlabel("cluster size: metastable $i$ and crystal $n$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ cluster population  [m$^{-3}$]", fontsize=12)
    ax.set_title(f"Overlaid cluster population histograms at t = {unit}", fontsize=13)
    ax.set_xlim(0.5, size_max + 0.5); ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    return fig


def plot_crystal_marginal_timeseries(I, element="", n_curves=6):
    t = I["time_s"]; M = I["params"].max_size; crit = I["crit"]
    n_plot = max(30, M - 4)
    idx = np.unique(np.geomspace(1, len(t) - 1, n_curves).astype(int))
    times = t[idx]
    norm = LogNorm(vmin=times.min(), vmax=times.max()); cmap = cm.plasma
    nn = np.arange(1, n_plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    for j in idx:
        c = mc.restore_actual_concentration(I["y"][:, j], I["w"], I["c1"], I["s_co"], I["params"])
        m = _marg_n(c, I["pairs"], M)[1:n_plot]
        ax.semilogy(nn, np.maximum(m, 1e-300), lw=2.0, color=cmap(norm(t[j])))
    ax.axvline(crit["n_star"], color="#2ca02c", ls="--", lw=1.6, label=f"$n^*={crit['n_star']}$")
    ax.set_xlabel("crystal cluster size $n$", fontsize=12)
    ax.set_ylabel(r"crystal-size marginal density $\sum_i c(i,n,t)$  [m$^{-3}$]", fontsize=12)
    ax.set_title("Time-dependent crystal cluster-size distribution", fontsize=13)
    ax.set_ylim(1e-210, 1e34); ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower left")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="time (s)")
    fig.tight_layout()
    return fig


def plot_noniso(B, element=""):
    has_stat = "J_d_stat" in B
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_d"], 1e-300), "-", color="#1f77b4", lw=2, label=r"$J_d$ (master eq.)")
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_com"], 1e-300), "-", color="#d62728", lw=2, label=r"$J_{com}$ (master eq.)")
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_c"], 1e-300), "-", color="#2ca02c", lw=1.5, label=r"$J_c$ (master eq.)")
    if has_stat:
        ax.semilogy(B["t"] * 1e3, np.maximum(B["J_d_stat"], 1e-300), "--", color="#1f77b4", lw=1.3, alpha=0.7, label=r"$J_d$ instantaneous stationary")
        ax.semilogy(B["t"] * 1e3, np.maximum(B["J_com_stat"], 1e-300), "--", color="#d62728", lw=1.3, alpha=0.7, label=r"$J_{com}$ instantaneous stationary")
    ax.set_xlabel("time (ms)"); ax.set_ylabel(r"nucleation rate (m$^{-3}$s$^{-1}$)")
    ax.set_title(f"Non-isothermal quench — {element}\n{B['T'][0]:.0f} K $\\to$ {B['T'][-1]:.0f} K over {B['t'][-1]*1e3:.3g} ms")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
    axT = ax.twinx()
    axT.plot(B["t"] * 1e3, B["T"], "--", color="gray", lw=1.4, alpha=0.7)
    axT.set_ylabel("T (K)", color="gray"); axT.tick_params(axis="y", colors="gray")
    fig.tight_layout()
    return fig


def plot_quench_population_histograms(B, t_target=5e-3, element="", size_max=40):
    """Overlaid metastable/crystal size populations at one instant during the quench."""
    t = B["t"]; j = int(np.argmin(np.abs(t - t_target)))
    Ni = B["N_i"][j]; Nn = B["N_n"][j]; Tj = B["T"][j]
    s = np.arange(1, size_max + 1)
    lNi = np.log10(np.maximum(Ni[1:size_max + 1], 1e-300))
    lNn = np.log10(np.maximum(Nn[1:size_max + 1], 1e-300))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(s, lNi, width=1.0, align="center", color="#9ecae1", alpha=0.65, edgecolor="#3182bd",
           lw=0.4, label=r"metastable-size population $N_i=\sum_n c(i,n)$")
    ax.bar(s, lNn, width=1.0, align="center", color="#fcbba1", alpha=0.65, edgecolor="#de2d26",
           lw=0.4, label=r"crystal-size population $N_n=\sum_i c(i,n)$")
    ax.set_xlabel("cluster size: metastable $i$ and crystal $n$", fontsize=12)
    ax.set_ylabel(r"$\log_{10}$ cluster population  [m$^{-3}$]", fontsize=12)
    ax.set_title(f"Cluster population during quench — t = {t[j]*1e3:.2f} ms  (T = {Tj:.0f} K)", fontsize=13)
    ax.set_xlim(0.5, size_max + 0.5); ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    return fig


def plot_quench_crystal_marginal(B, element="", n_curves=8):
    """Crystal-size marginal vs n at several quench instants, coloured by temperature."""
    t = B["t"]; T = B["T"]; Nn = np.asarray(B["N_n"])
    M = Nn.shape[1] - 2
    n_plot = max(30, M - 4)
    npts = len(t)
    idx = np.unique(np.linspace(0, npts - 1, n_curves).astype(int))
    norm = Normalize(vmin=float(T.min()), vmax=float(T.max())); cmap = cm.coolwarm
    nn = np.arange(1, n_plot)
    fig, ax = plt.subplots(figsize=(11, 7))
    for j in idx:
        ax.semilogy(nn, np.maximum(Nn[j][1:n_plot], 1e-300), lw=2.0, color=cmap(norm(T[j])))
    ns_hot, ns_cold = int(B["n_star"][0]), int(B["n_star"][-1])
    ax.axvline(ns_cold, color="#1f77b4", ls="--", lw=1.4, label=f"$n^*$ at {T[-1]:.0f} K = {ns_cold}")
    if ns_hot != ns_cold:
        ax.axvline(ns_hot, color="#b2182b", ls="--", lw=1.4, label=f"$n^*$ at {T[0]:.0f} K = {ns_hot}")
    ax.set_xlabel("crystal cluster size $n$", fontsize=12)
    ax.set_ylabel(r"crystal-size marginal density $\sum_i c(i,n,t)$  [m$^{-3}$]", fontsize=12)
    ax.set_title(f"Crystal cluster-size distribution during the quench — {element}", fontsize=13)
    ax.set_ylim(bottom=1e-40); ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax); cb.set_label("temperature (K)")
    fig.tight_layout()
    return fig
