# Theoretical Background

This repository connects a two-step nucleation calculation to a real-space phase-field growth
model. The two parts solve different pieces of the same physical process:

- The master-equation model estimates when nuclei appear and through which pathway.
- The phase-field model evolves the spatial growth and competition after post-critical nuclei are
  introduced.

## 1. From CNT to Two-Step Nucleation

Classical nucleation theory usually describes a nucleus by one size coordinate. In that picture,
the system crosses a free-energy barrier directly from the parent phase to the final crystal. The
two-step model used here instead allows an intermediate amorphous cluster to form first. The state
of a composite cluster is therefore described by two integers:

| Symbol | Meaning |
|---|---|
| `i` | total metastable/amorphous monomers in the composite cluster |
| `n` | crystal monomers inside that metastable cluster |

The valid state space is triangular: `1 ≤ n ≤ i`. This allows three rates to be compared in the
same framework:

| Rate | Pathway |
|---|---|
| `J_d` | parent → amorphous/metastable cluster |
| `J_com` | crystal nucleation inside the amorphous cluster |
| `J_c` | direct parent → crystal nucleation |

For the Fe 160 K case in the notebooks, `J_d` and `J_com` define the operative two-step pathway,
while `J_c` is negligible on the same scale.

## 2. Master Equation and Attachment Kinetics

The population of each cluster state evolves by a master equation over the `(i, n)` grid. Neighboring
states are connected by attachment and detachment events. Solid-state mobility enters through the
Turnbull-Fisher attachment frequency,

```text
Ω = 24D / λ²
```

where `D` is a diffusion coefficient and `λ` is a jump length. The notebook integrates the population
distribution in time, then extracts stationary nucleation rates, induction times, critical sizes, and
supersaturation-dependent quantities.

The key practical output of this stage is not just a single rate. It is a consistent parameter set:

```text
J_d, J_com, J_c, θ_d, θ_com, critical cluster sizes, and population maps
```

These quantities define both the statistics and the timing of seeding in the phase-field stage.

## 3. Coupling to Phase Field

The phase-field notebook uses two non-conserved order parameters:

| Field | Phase represented |
|---|---|
| `η_m` | amorphous/metastable phase |
| `η_c` | crystal/BCC phase |

The parent/FCC phase is represented by the absence of both transformed order parameters. Growth is
then governed by Allen-Cahn dynamics: the free-energy functional supplies local driving forces,
double-well barriers, and gradient penalties, and the fields evolve downhill in free energy.

The handoff from nucleation theory to phase field is done with Simmons-style explicit seeding:

```text
P = 1 - exp(-J ΔV Δt)
```

where `J` is the relevant nucleation rate, `ΔV` is the cell volume, and `Δt` is the physical time
step. The two-step pathway is enforced by gating:

- amorphous seeds are allowed after `θ_d` in parent regions;
- crystal seeds are allowed after `θ_com` only inside regions that are already amorphous.

This means the phase-field model does not invent the nucleation pathway. It receives the pathway and
timing from the master-equation calculation, then resolves how transformed regions grow, overlap, and
compete in space.

## 4. Why the Figure Outputs Matter

The exported figures are grouped by the same logic:

- `figures/01_nucleation_engine/` shows rates, induction behavior, work-of-formation landscapes, and
  population/concentration maps from the master-equation notebook.
- `figures/02_phase_field_coupling/` shows the phase-field setup and spatial two-step growth outputs.

Together, the figures provide a quick check that the repository contains both the parameter-producing
stage and the spatial growth stage.
