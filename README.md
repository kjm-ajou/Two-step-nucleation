# Two-Step Nucleation Calculator (unary metals)

A web app that computes **two-step nucleation** (parent → metastable → crystal) for any
single-element metal. You enter physical parameters; the app nondimensionalizes them, fits the
temperature dependence, and runs a Turnbull–Fisher composite-cluster master equation to produce:

- isothermal nucleation rates `J_d`, `J_com`, `J_c` and induction times,
- the cluster population distribution: the nucleation work surface, overlaid metastable-size and
  crystal-size population histograms, and the time-dependent crystal cluster-size distribution,
- an optional **non-isothermal quench** `T(t)`: the same cluster-population figures (population
  histograms and the temperature-resolved crystal cluster-size distribution) plus nucleation
  rates that capture transient/incubation lag.

The defaults reproduce a validated Fe baseline
(`J_d ≈ 4.31×10³⁵`, `J_com ≈ 5.27×10³¹`, `J_c ≈ 1.97×10¹⁶ m⁻³s⁻¹`).

## How it works (two parts)

The browser shows the input form and figures; the heavy computation (SciPy sparse + stiff BDF
solver) runs on a Python server. The browser sends parameters, the server computes and returns the
figures. That is why this is deployed as a server app, not a static page.

## Files

```
app.py               # Streamlit UI (the only file with UI code)
app_logic.py         # build/validate inputs, run isothermal & quench, make figures
general_support.py   # physical→dimensionless, T-dependence fitting, runners
general_plots.py     # matplotlib figures
model_core.py        # the validated Turnbull–Fisher two-step master-equation engine
requirements.txt
```

## Run locally

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Edit the parameters and press **Run**.
(First run with a quench takes ~30–60 s — the master equation is stiff and is solved with BDF.)

## Deploy for free (get a public URL)

### Option A — Streamlit Community Cloud (simplest)
1. Put these files in a **public GitHub repository**.
2. Go to https://share.streamlit.io , sign in with GitHub, click **New app**.
3. Select your repo/branch and set the main file to `app.py`. Click **Deploy**.
4. You get a public URL like `https://<your-app>.streamlit.app`.

Streamlit Cloud installs `requirements.txt` automatically. The free tier has limited CPU/RAM, so
keep the quench segment count modest (the default 15 is fine).

### Option B — Hugging Face Spaces
1. Create a new **Space** at https://huggingface.co/new-space , SDK = **Streamlit**.
2. Upload all the files above (or push with git). Spaces installs `requirements.txt` automatically.
3. The Space serves a public URL.

## Usage notes

- **Units**: `dmu` in eV/atom, `sigma` in J/m², `D` in m²/s, atomic volume in Å³/atom, jump lengths
  in Å. Conversion to SI is automatic.
- **Temperature dependence**: give ≥ 2 rows in the parameter table (linear fit for 2, quadratic for
  ≥ 3; `D` is Arrhenius-fitted as ln D vs 1/T). For real work use ~5–6 temperatures spanning the
  quench. The 200 K row in the example is an illustrative placeholder — replace it.
- **Grid bound** `max_size`: keep it well above the largest critical size; the app checks this and
  warns. Lower supersaturation (high T or near a transition) raises the critical size.
- **Assumptions**: the monomer is pinned to its equilibrium concentration at each temperature (no
  depletion); the quench tolerance is tuned for the dominant channels `J_d`/`J_com` (`J_c` needs a
  much tighter tolerance and is usually negligible).

## Method references
- D. Kashchiev, *J. Cryst. Growth* **530**, 125300 (2020) — two-step composite-cluster model.
- K. F. Kelton, A. L. Greer, *Nucleation in Condensed Matter*, Pergamon (2010), Ch. 2 (CNT).
