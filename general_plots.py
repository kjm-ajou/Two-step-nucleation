"""Plotting helpers for the general unary-metal two-step nucleation notebook."""
import numpy as np
import matplotlib.pyplot as plt
import model_core as mc  # in the notebook this is rebound to sys.modules['__main__']


def _crystal_marginal(c_flat, pairs, max_size):
    """sum over i of c(i,n) for each crystal size n."""
    m = np.zeros(max_size + 2)
    np.add.at(m, pairs[:, 1], c_flat)
    return m


def plot_iso(I, max_size, element=""):
    pairs, w, c1, sco, params = I["pairs"], I["w"], I["c1"], I["s_co"], I["params"]
    y, t, crit = I["y"], I["time_s"], I["crit"]
    fig = plt.figure(figsize=(14.5, 8.4))
    fig.suptitle(f"Isothermal two-step nucleation — {element}  (T = {params.temperature_K:.0f} K)",
                 fontsize=13, y=0.99)

    # (a) rates vs time
    ax = fig.add_subplot(2, 3, 1)
    for nm, col in [("J_d", "#1f77b4"), ("J_com", "#d62728"), ("J_c", "#2ca02c")]:
        ax.loglog(t[1:], np.maximum(I["rates_t"][nm][1:], 1e-300), color=col, lw=1.8, label=nm)
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$J$ (m$^{-3}$s$^{-1}$)")
    ax.set_title("(a) nucleation rates"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (b) number densities
    ax = fig.add_subplot(2, 3, 2)
    for nm, col in [("N_d", "#1f77b4"), ("N_com", "#d62728"), ("N_c", "#2ca02c")]:
        ax.loglog(t[1:], np.maximum(I["dens_t"][nm][1:], 1e-300), color=col, lw=1.8, label=nm)
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$N$ (m$^{-3}$)")
    ax.set_title("(b) cumulative number density"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (c) work-of-formation surface
    ax = fig.add_subplot(2, 3, 3)
    Wg = mc.flat_to_triangular_grid(w, pairs, max_size)
    im = ax.imshow(Wg, origin="lower", aspect="auto", cmap="viridis",
                   extent=[1, max_size, 1, max_size], vmin=np.nanmin(Wg),
                   vmax=np.nanpercentile(Wg, 99))
    ax.plot(crit["i_star"], 1, "rx", ms=8); ax.plot(crit["i_co_star"], crit["i_co_star"], "wx", ms=8)
    ax.set_xlabel("i (metastable size)"); ax.set_ylabel("n (crystal size)")
    ax.set_title(r"(c) work surface $w_{i,n}/k_BT$"); fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) final concentration map
    ax = fig.add_subplot(2, 3, 4)
    Cg = mc.flat_to_triangular_grid(I["log10c_final"], pairs, max_size)
    im = ax.imshow(Cg, origin="lower", aspect="auto", cmap="magma",
                   extent=[1, max_size, 1, max_size],
                   vmin=np.nanpercentile(Cg, 5), vmax=np.nanmax(Cg))
    ax.set_xlabel("i"); ax.set_ylabel("n")
    ax.set_title(r"(d) $\log_{10} c(i,n)$ at $t_{end}$"); fig.colorbar(im, ax=ax, fraction=0.046)

    # (e) crystal-size marginal at several times
    ax = fig.add_subplot(2, 3, 5)
    idxs = np.unique(np.linspace(1, len(t) - 1, 5).astype(int))
    nmax = min(max_size, max(40, 2 * crit["n_star"]))
    nn = np.arange(nmax + 1)
    for j in idxs:
        c = mc.restore_actual_concentration(y[:, j], w, c1, sco, params)
        m = _crystal_marginal(c, pairs, max_size)[: nmax + 1]
        ax.semilogy(nn, np.maximum(m, 1e-300), lw=1.4, label=f"t={t[j]:.1e}s")
    ax.axvline(crit["n_star"], color="k", ls=":", lw=1, label=f"n*={crit['n_star']}")
    ax.set_xlabel("crystal size n"); ax.set_ylabel(r"$\sum_i c(i,n)$ (m$^{-3}$)")
    ax.set_title("(e) crystal-size marginal"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    ax.set_ylim(bottom=max(1e-10, np.nanmin([1e10])))

    # (f) population centroid trajectory
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(Wg, origin="lower", aspect="auto", cmap="Greys",
              extent=[1, max_size, 1, max_size], alpha=0.5,
              vmin=np.nanmin(Wg), vmax=np.nanpercentile(Wg, 95))
    sc = ax.scatter(I["cen_i"], I["cen_n"], c=np.log10(np.maximum(t, 1e-13)),
                    cmap="plasma", s=18, zorder=5)
    ax.plot(crit["i_star"], 1, "rx", ms=8)
    ax.set_xlabel("i"); ax.set_ylabel("n")
    ax.set_title("(f) population centroid path")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046); cb.set_label(r"$\log_{10} t$")
    lim = min(max_size, max(40, 2 * crit["n_star"]))
    ax.set_xlim(1, lim); ax.set_ylim(1, lim)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_noniso(B, element=""):
    has_stat = "J_d_stat" in B
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_d"], 1e-300), "-", color="#1f77b4", lw=2,
                label=r"$J_d$ (master eq.)")
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_com"], 1e-300), "-", color="#d62728", lw=2,
                label=r"$J_{com}$ (master eq.)")
    ax.semilogy(B["t"] * 1e3, np.maximum(B["J_c"], 1e-300), "-", color="#2ca02c", lw=1.5,
                label=r"$J_c$ (master eq.)")
    if has_stat:
        ax.semilogy(B["t"] * 1e3, np.maximum(B["J_d_stat"], 1e-300), "--", color="#1f77b4",
                    lw=1.3, alpha=0.7, label=r"$J_d$ instantaneous stationary")
        ax.semilogy(B["t"] * 1e3, np.maximum(B["J_com_stat"], 1e-300), "--", color="#d62728",
                    lw=1.3, alpha=0.7, label=r"$J_{com}$ instantaneous stationary")
    ax.set_xlabel("time (ms)"); ax.set_ylabel(r"nucleation rate (m$^{-3}$s$^{-1}$)")
    ax.set_title(f"Non-isothermal quench — {element}\n"
                 f"{B['T'][0]:.0f} K $\\to$ {B['T'][-1]:.0f} K over {B['t'][-1]*1e3:.3g} ms")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="best")
    axT = ax.twinx()
    axT.plot(B["t"] * 1e3, B["T"], "--", color="gray", lw=1.4, alpha=0.7)
    axT.set_ylabel("T (K)", color="gray"); axT.tick_params(axis="y", colors="gray")
    fig.tight_layout()
    return fig
