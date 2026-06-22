import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import app_logic as L

st.set_page_config(page_title="Two-Step Nucleation Calculator", layout="wide")

st.title("Two-Step Nucleation Calculator — unary metals")
st.markdown(
    "Compute **two-step nucleation** (parent → metastable → crystal) for any single-element metal. "
    "Enter physical parameters; the app nondimensionalizes them, fits the temperature dependence, and "
    "runs a Turnbull–Fisher master equation for the isothermal nucleation rates, the cluster population "
    "distribution, and an optional non-isothermal quench. The defaults reproduce a validated Fe baseline "
    "(J_d ≈ 4.31×10³⁵, J_com ≈ 5.27×10³¹, J_c ≈ 1.97×10¹⁶ m⁻³s⁻¹)."
)

with st.expander("What the inputs mean (units)", expanded=False):
    st.markdown(
        "- **dmu_mo / dmu_cm** — chemical-potential drop per atom [eV]: parent→metastable, "
        "metastable→crystal (parent→crystal is their sum).\n"
        "- **sigma_mo / sigma_cm / sigma_co** — interface energies [J/m²]: M\\|O, C\\|M, C\\|O.\n"
        "- **D_f / D_g / D_k** — diffusion coefficients [m²/s]: attachment to the metastable phase, "
        "in-cluster M→C conversion, direct crystal attachment (each Arrhenius-fitted vs T).\n"
        "- **Atomic volume** [Å³/atom], **jump lengths** [Å], **max_size** = triangular-grid bound "
        "(keep ≥ ~5× the largest critical size).\n"
        "- Give **≥ 2 temperature rows** so a T-dependence can be fitted (linear for 2 rows, quadratic "
        "for ≥ 3; D uses ln D vs 1/T). The 200 K row in the example is an illustrative placeholder — "
        "replace it with your own data."
    )

with st.form("inputs"):
    st.subheader("1. System")
    c = st.columns(5)
    element = c[0].text_input("Element / system", L.DEFAULT_SCALARS["element"])
    vol = c[1].number_input("Atomic volume (Å³/atom)", value=L.DEFAULT_SCALARS["atomic_volume_A3"],
                            min_value=0.0, format="%.4f")
    jf = c[2].number_input("Jump length f (Å)", value=L.DEFAULT_SCALARS["jump_f_A"], min_value=0.0, format="%.4f")
    jg = c[3].number_input("Jump length g (Å)", value=L.DEFAULT_SCALARS["jump_g_A"], min_value=0.0, format="%.4f")
    jk = c[4].number_input("Jump length k (Å)", value=L.DEFAULT_SCALARS["jump_k_A"], min_value=0.0, format="%.4f")
    max_size = int(st.number_input("Grid bound max_size", value=L.DEFAULT_SCALARS["max_size"],
                                   min_value=30, step=10))

    st.subheader("2. Temperature-dependent parameters")
    st.caption("dmu in eV/atom · sigma in J/m² · D in m²/s. Add rows for more anchors (≥ 2 needed).")
    anchors = st.data_editor(L.DEFAULT_ANCHORS, num_rows="dynamic", key="anchors")

    st.subheader("3. Run settings")
    st.markdown("**Isothermal**")
    ca = st.columns(2)
    T_iso = ca[0].number_input("Isothermal T (K)", value=160.0, min_value=1.0)
    hist_time = ca[1].number_input("Population-histogram time (s)", value=5e-6, min_value=0.0, format="%.2e")
    st.markdown("**Quench**")
    do_quench = st.checkbox("Run a quench (cluster population + rates)", value=True)
    cb = st.columns(5)
    T_hot = cb[0].number_input("From (K)", value=200.0, min_value=1.0)
    T_cold = cb[1].number_input("To (K)", value=160.0, min_value=1.0)
    total_time = cb[2].number_input("Duration (s)", value=1e-2, min_value=0.0, format="%.4g")
    n_seg = int(cb[3].number_input("Segments", value=15, min_value=2, max_value=200, step=1))
    quench_hist_time = cb[4].number_input("Snapshot time (s)", value=5e-3, min_value=0.0, format="%.2e")

    submitted = st.form_submit_button("Run", type="primary")

if not submitted:
    st.info("Fe defaults are loaded. Edit the parameters above and press **Run**.")
    st.stop()

# ---- build & validate ----
scalars = dict(element=element, atomic_volume_A3=vol, jump_f_A=jf, jump_g_A=jg, jump_k_A=jk, max_size=max_size)
user, df = L.build_user(scalars, anchors)
errs = L.validate(user, df)
if errs:
    for e in errs:
        st.error(e)
    st.stop()

# ---- nondimensional summary + grid check ----
T_list = [T_cold, 0.5 * (T_cold + T_hot), T_hot, T_iso] if do_quench else [T_iso]
nd = L.nondim_summary(user, T_iso)
gdf, worst, suggested, adequate = L.grid_check(user, T_list)

st.subheader(f"Parameters at {T_iso:.0f} K (after nondimensionalization)")
cc = st.columns(3)
cc[0].metric("s_co", f"{nd['s_co']:.3f}")
cc[0].metric("s_cm", f"{nd['s_cm']:.3f}")
cc[0].metric("s_mo", f"{nd['s_mo']:.3f}")
cc[1].metric("γ_mo", f"{nd['gamma_mo']:.2f}")
cc[1].metric("γ_cm", f"{nd['gamma_cm']:.2f}")
cc[1].metric("γ_co", f"{nd['gamma_co']:.2f}")
cc[2].metric("Q(D_f) eV", f"{nd['Q_Df_eV']:.3f}")
cc[2].metric("Q(D_g) eV", f"{nd['Q_Dg_eV']:.3f}")
cc[2].metric("Q(D_k) eV", f"{nd['Q_Dk_eV']:.3f}")

st.markdown("**Grid-bound check** — critical sizes across the temperature range:")
st.dataframe(gdf, hide_index=True)
if adequate:
    st.success(f"max_size = {user['max_size']} is adequate (≥ 3× the largest critical size, {worst}).")
else:
    st.warning(f"Largest critical size is {worst}; raise max_size to ≥ {suggested} for a safe margin.")

# ---- isothermal ----
st.subheader(f"Isothermal results at {T_iso:.0f} K")
with st.spinner("Solving the master equation (isothermal)…"):
    rates, crit, I = L.run_isothermal(user, T_iso)
m = st.columns(3)
m[0].metric("J_d  (amorphous)", f"{rates['J_d']:.3e}")
m[1].metric("J_com  (crystal in amorphous, 2S)", f"{rates['J_com']:.3e}")
m[2].metric("J_c  (direct crystal)", f"{rates['J_c']:.3e}")
st.caption(f"rates in m⁻³ s⁻¹ · critical sizes  i*={crit['i_star']},  n*={crit['n_star']},  i_co*={crit['i_co_star']}")

figs = L.figures_isothermal(I, user["element"], hist_time_s=hist_time)
st.pyplot(figs["rates"]); plt.close(figs["rates"])
st.markdown("**Nucleation work surface** — the 2D landscape; the two-step path skirts the high direct-crystal barrier.")
st.pyplot(figs["work"]); plt.close(figs["work"])
st.markdown("**Cluster population** — metastable-size ($N_i$) vs crystal-size ($N_n$) marginals at the chosen time.")
st.pyplot(figs["hist"]); plt.close(figs["hist"])
st.markdown("**Crystal cluster-size distribution over time** — builds toward steady state; $n^*$ marked.")
st.pyplot(figs["marginal"]); plt.close(figs["marginal"])

# ---- non-isothermal ----
if do_quench:
    st.subheader(f"Non-isothermal quench  {T_hot:.0f} K → {T_cold:.0f} K")
    with st.spinner(f"Stepping the quench ({n_seg} segments)… this can take ~30–60 s"):
        qinfo, B = L.run_quench(user, T_hot, T_cold, total_time, n_seg)
    if qinfo["n_fail"]:
        st.error(f"{qinfo['n_fail']} segment(s) failed to converge — try fewer segments or a larger max_size.")
    else:
        st.success(f"All {n_seg} segments converged ({qinfo['n_points']} output points).")
    figs_q = L.figures_quench(B, user["element"], hist_time_s=quench_hist_time)
    st.markdown("**Cluster population during the quench** — metastable-size ($N_i$) vs crystal-size ($N_n$) at the chosen instant.")
    st.pyplot(figs_q["hist"]); plt.close(figs_q["hist"])
    st.markdown("**Crystal cluster-size distribution as it cools** — each curve is one instant, coloured by temperature; $n^*$ shifts with T.")
    st.pyplot(figs_q["marginal"]); plt.close(figs_q["marginal"])
    st.markdown("**Nucleation rates** — solid = master equation (captures transient/incubation lag); dashed = instantaneous stationary. A fast quench makes them diverge.")
    st.pyplot(figs_q["rate"]); plt.close(figs_q["rate"])

st.divider()
st.caption(
    "Assumptions: monomer pinned to equilibrium at each T (no depletion); quench atol tuned for J_d/J_com "
    "(J_c needs tighter tolerance). Fits are only as good as the anchor points — use ~5–6 temperatures "
    "spanning the quench for real work."
)
