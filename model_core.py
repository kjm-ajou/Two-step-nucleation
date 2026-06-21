# Embedded model core copied into this notebook.
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp
import math
from types import SimpleNamespace
from typing import Dict, Iterable, List, Literal, Tuple, cast

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.integrate import odeint, solve_ivp
except ImportError:  # pragma: no cover - environment-dependent
    sp = None
    spla = None
    odeint = None
    solve_ivp = None


_FLOAT_MAX = np.finfo(float).max
_LOG_FLOAT_MAX = math.log(_FLOAT_MAX)
_LOG_FLOAT_MIN = math.log(np.finfo(float).tiny)
_LOG_FLUX_CLIP = 700.0


@dataclass(frozen=True)
class Paper2020Figure4Params:
    s_cm: float = 0.55
    gamma_mo: float = 13.0
    gamma_cm: float = 2.5
    gamma_co: float = 15.5


@dataclass(frozen=True)
class Paper2022Params:
    temperature_K: float = 225.0
    s_cm: float = 0.5
    gamma_mo: float = 12.8
    gamma_cm: float = 2.6
    gamma_co: float = 15.4
    f0: float = 1.0e5
    g0: float = 2.0e7
    q_split: float = 0.5
    c1_base: float = 1.6e21
    max_size: int = 240


@dataclass(frozen=True)
class ChannelClosure:
    family: Literal["legacy_water", "activated_interfacial", "turnbull_fisher"]
    prefactor_s: float
    area_exponent: float = 2.0 / 3.0
    activation_barrier_over_kbt: float = 0.0
    driving_exponential: float = 0.0
    split_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.family != "activated_interfacial":
            return
        if self.driving_exponential != 0.0:
            raise ValueError(
                "activated_interfacial closures require neutral driving_exponential=0.0"
            )
        if self.split_fraction != 1.0:
            raise ValueError(
                "activated_interfacial closures require neutral split_fraction=1.0"
            )


@dataclass(frozen=True)
class TwoStepClosureSet:
    f_channel: ChannelClosure
    g_channel: ChannelClosure
    k_channel: ChannelClosure
    reference_rate_s: float | None = None


def _resolve_closure_set(
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None,
) -> TwoStepClosureSet:
    return closure_set or make_water_two_step_closure_set(params)


def _resolve_reference_rate(
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None,
) -> float:
    closures = _resolve_closure_set(params, closure_set)
    if closures.reference_rate_s is not None:
        return float(closures.reference_rate_s)
    return float(params.f0)


def diffusion_turnbull_fisher_frequency(
    diffusion_m2_s: float,
    jump_length_m: float,
) -> float:
    if diffusion_m2_s <= 0.0:
        raise ValueError("diffusion_m2_s must be positive")
    if jump_length_m <= 0.0:
        raise ValueError("jump_length_m must be positive")
    return 24.0 * float(diffusion_m2_s) / float(jump_length_m) ** 2


def make_diffusion_turnbull_fisher_closure_set(
    *,
    f_diffusion_m2_s: float,
    f_jump_length_m: float,
    g_diffusion_m2_s: float,
    g_jump_length_m: float,
    k_diffusion_m2_s: float,
    k_jump_length_m: float,
    area_exponent: float = 2.0 / 3.0,
) -> TwoStepClosureSet:
    f_prefactor = diffusion_turnbull_fisher_frequency(f_diffusion_m2_s, f_jump_length_m)
    g_prefactor = diffusion_turnbull_fisher_frequency(g_diffusion_m2_s, g_jump_length_m)
    k_prefactor = diffusion_turnbull_fisher_frequency(k_diffusion_m2_s, k_jump_length_m)
    return TwoStepClosureSet(
        f_channel=ChannelClosure(
            family="turnbull_fisher",
            prefactor_s=f_prefactor,
            area_exponent=area_exponent,
        ),
        g_channel=ChannelClosure(
            family="turnbull_fisher",
            prefactor_s=g_prefactor,
            area_exponent=area_exponent,
        ),
        k_channel=ChannelClosure(
            family="turnbull_fisher",
            prefactor_s=k_prefactor,
            area_exponent=area_exponent,
        ),
        reference_rate_s=f_prefactor,
    )


@lru_cache(maxsize=None)
def make_water_two_step_closure_set(params: Paper2022Params) -> TwoStepClosureSet:
    return TwoStepClosureSet(
        f_channel=ChannelClosure(
            family="legacy_water",
            prefactor_s=params.f0,
            area_exponent=2.0 / 3.0,
            activation_barrier_over_kbt=0.0,
            driving_exponential=1.0,
            split_fraction=1.0 - params.q_split,
        ),
        g_channel=ChannelClosure(
            family="legacy_water",
            prefactor_s=params.g0,
            area_exponent=2.0 / 3.0,
            activation_barrier_over_kbt=0.0,
            driving_exponential=1.0,
            split_fraction=1.0,
        ),
        k_channel=ChannelClosure(
            family="legacy_water",
            prefactor_s=params.f0,
            area_exponent=2.0 / 3.0,
            activation_barrier_over_kbt=0.0,
            driving_exponential=1.0,
            split_fraction=params.q_split,
        ),
    )


def paper2020_figure4_curves(
    s_co_values: np.ndarray,
    params: Paper2020Figure4Params | None = None,
) -> Dict[str, np.ndarray]:
    p = params or Paper2020Figure4Params()
    s_co_values = np.asarray(s_co_values, dtype=float)
    s_mo = s_co_values - p.s_cm

    with np.errstate(divide="ignore", invalid="ignore"):
        i_star = (2.0 * p.gamma_mo / (3.0 * s_mo)) ** 3
        n_star = np.full_like(s_co_values, (2.0 * p.gamma_cm / (3.0 * p.s_cm)) ** 3)
        i_co_star = (2.0 * p.gamma_co / (3.0 * s_co_values)) ** 3

        w_star = (
            4.0 * p.gamma_mo**3 / (27.0 * s_mo**2)
            + 4.0 * p.gamma_cm**3 / (27.0 * p.s_cm**2)
        )
        w_mo_star = 4.0 * p.gamma_mo**3 / (27.0 * s_mo**2)
        w_cm_star = np.full_like(s_co_values, 4.0 * p.gamma_cm**3 / (27.0 * p.s_cm**2))
        w_co_star = 4.0 * p.gamma_co**3 / (27.0 * s_co_values**2)

    return {
        "s_co": s_co_values,
        "s_mo": s_mo,
        "i_star": i_star,
        "n_star": n_star,
        "i_co_star": i_co_star,
        "w_star": w_star,
        "w_mo_star": w_mo_star,
        "w_cm_star": w_cm_star,
        "w_co_star": w_co_star,
    }


def stationary_rate_table_log10() -> List[Dict[str, float]]:
    return [
        {"s_co": 2.0, "J_c": -25.332, "J_c_1S": -25.118, "J_c_d": -24.077, "J_d": -27.312, "J_d_1S": -27.292, "J_com": -24.327, "J_d_plus_com": -24.326, "J_d_plus_com_plus_c": -24.286},
        {"s_co": 2.2, "J_c": -14.889, "J_c_1S": -14.836, "J_c_d": -13.873, "J_d": -13.943, "J_d_1S": -13.926, "J_com": -13.654, "J_d_plus_com": -13.474, "J_d_plus_com_plus_c": -13.457},
        {"s_co": 2.5, "J_c": -3.177, "J_c_1S": -3.753, "J_c_d": -2.101, "J_d": -0.849, "J_d_1S": -0.840, "J_com": -0.933, "J_d_plus_com": -0.588, "J_d_plus_com_plus_c": -0.587},
        {"s_co": 3.0, "J_c": 8.498, "J_c_1S": 7.953, "J_c_d": 9.745, "J_d": 11.517, "J_d_1S": 11.521, "J_com": 11.411, "J_d_plus_com": 11.769, "J_d_plus_com_plus_c": 11.769},
        {"s_co": 3.5, "J_c": 14.630, "J_c_1S": 15.096, "J_c_d": 16.255, "J_d": 18.330, "J_d_1S": 18.334, "J_com": 18.220, "J_d_plus_com": 18.580, "J_d_plus_com_plus_c": 18.580},
        {"s_co": 4.0, "J_c": 17.561, "J_c_1S": 19.809, "J_c_d": 20.170, "J_d": 22.522, "J_d_1S": 22.528, "J_com": 22.410, "J_d_plus_com": 22.771, "J_d_plus_com_plus_c": 22.771},
        {"s_co": 4.5, "J_c": 17.883, "J_c_1S": 23.107, "J_c_d": 22.648, "J_d": 25.316, "J_d_1S": 25.326, "J_com": 25.201, "J_d_plus_com": 25.563, "J_d_plus_com_plus_c": 25.563},
        {"s_co": 5.0, "J_c": 15.525, "J_c_1S": 25.529, "J_c_d": 24.220, "J_d": 27.290, "J_d_1S": 27.312, "J_com": 27.173, "J_d_plus_com": 27.537, "J_d_plus_com_plus_c": 27.537},
        {"s_co": 5.5, "J_c": 14.418, "J_c_1S": 27.377, "J_c_d": 25.100, "J_d": 28.754, "J_d_1S": 28.794, "J_com": 28.634, "J_d_plus_com": 28.999, "J_d_plus_com_plus_c": 28.999},
        {"s_co": 6.0, "J_c": 15.643, "J_c_1S": 28.834, "J_c_d": 25.223, "J_d": 29.880, "J_d_1S": 29.947, "J_com": 29.756, "J_d_plus_com": 30.124, "J_d_plus_com_plus_c": 30.124},
    ]


def delay_time_table_microseconds() -> List[Dict[str, float]]:
    return [
        {"s_co": 2.0, "theta_c": -4.79, "theta_c_1S": 19.3, "theta_c_d": -1.95e13, "theta_d": 29.8, "theta_d_1S": 30.3, "theta_com": 40.4, "theta_d_plus_com": 40.4, "theta_d_plus_com_plus_c": 36.3},
        {"s_co": 2.2, "theta_c": -1.99, "theta_c_1S": 12.1, "theta_c_d": -1.16e7, "theta_d": 17.0, "theta_d_1S": 17.6, "theta_com": 21.3, "theta_d_plus_com": 19.8, "theta_d_plus_com_plus_c": 19.0},
        {"s_co": 2.5, "theta_c": 10.5, "theta_c_1S": 6.49, "theta_c_d": -16.8, "theta_d": 8.09, "theta_d_1S": 8.33, "theta_com": 8.68, "theta_d_plus_com": 8.36, "theta_d_plus_com_plus_c": 8.36},
        {"s_co": 3.0, "theta_c": 7.00, "theta_c_1S": 2.47, "theta_c_d": 9.13, "theta_d": 2.64, "theta_d_1S": 2.69, "theta_com": 2.77, "theta_d_plus_com": 2.69, "theta_d_plus_com_plus_c": 2.70},
        {"s_co": 3.5, "theta_c": 2.16, "theta_c_1S": 0.971, "theta_c_d": 4.74, "theta_d": 1.10, "theta_d_1S": 1.11, "theta_com": 1.15, "theta_d_plus_com": 1.12, "theta_d_plus_com_plus_c": 1.12},
        {"s_co": 4.0, "theta_c": -32.9, "theta_c_1S": 0.340, "theta_c_d": 2.56, "theta_d": 0.402, "theta_d_1S": 0.404, "theta_com": 0.428, "theta_d_plus_com": 0.414, "theta_d_plus_com_plus_c": 0.413},
        {"s_co": 4.5, "theta_c": -1.05e4, "theta_c_1S": 0.145, "theta_c_d": 1.47, "theta_d": 0.155, "theta_d_1S": 0.155, "theta_com": 0.177, "theta_d_plus_com": 0.165, "theta_d_plus_com_plus_c": 0.164},
        {"s_co": 5.0, "theta_c": -2.12e8, "theta_c_1S": 0.0720, "theta_c_d": 0.886, "theta_d": 0.0602, "theta_d_1S": 0.0622, "theta_com": 0.0835, "theta_d_plus_com": 0.0703, "theta_d_plus_com_plus_c": 0.0701},
        {"s_co": 5.5, "theta_c": -7.08e10, "theta_c_1S": 0.0374, "theta_c_d": 0.552, "theta_d": 0.0189, "theta_d_1S": 0.0231, "theta_com": 0.0437, "theta_d_plus_com": 0.0296, "theta_d_plus_com_plus_c": 0.0294},
        {"s_co": 6.0, "theta_c": -3.79e10, "theta_c_1S": 0.0251, "theta_c_d": 0.335, "theta_d": 0.00911, "theta_d_1S": 0.0139, "theta_com": 0.0310, "theta_d_plus_com": 0.0185, "theta_d_plus_com_plus_c": 0.0184},
    ]


def triangular_state_pairs(max_size: int) -> List[Tuple[int, int]]:
    return [(i, n) for i in range(1, max_size) for n in range(1, i + 1)]


def triangular_index_maps(max_size: int) -> Tuple[Dict[Tuple[int, int], int], np.ndarray]:
    pairs = triangular_state_pairs(max_size)
    pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}
    return pair_to_index, np.array(pairs, dtype=int)


def w_total(i: int, n: int, s_co: float, params: Paper2022Params) -> float:
    s_mo = s_co - params.s_cm
    return (
        -s_mo * float(i)
        + params.gamma_mo * float(i) ** (2.0 / 3.0)
        - params.s_cm * float(n)
        + params.gamma_cm * float(n) ** (2.0 / 3.0)
    )


def make_w_and_c_arrays(
    s_co: float,
    params: Paper2022Params,
    state_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    c1 = params.c1_base * np.exp(s_co)
    w = np.array([w_total(int(i), int(n), s_co, params) for i, n in state_pairs], dtype=float)
    w11 = w_total(1, 1, s_co, params)
    log_c_eq = math.log(float(c1)) + w11 - w
    c_eq = _exp_from_log_clipped(log_c_eq)
    return w, c_eq, c1


def critical_sizes(s_co: float, params: Paper2022Params) -> Dict[str, int]:
    s_mo = s_co - params.s_cm
    i_star = int(np.ceil((2.0 * params.gamma_mo / (3.0 * s_mo)) ** 3))
    n_star = int(np.ceil((2.0 * params.gamma_cm / (3.0 * params.s_cm)) ** 3))
    i_co_star = int(np.ceil((2.0 * params.gamma_co / (3.0 * s_co)) ** 3))
    return {"i_star": i_star, "n_star": n_star, "i_co_star": i_co_star}


def _validate_stable_reconstruction_domain(
    s_co: float,
    params: Paper2022Params,
    critical: Dict[str, int] | None = None,
) -> Dict[str, int]:
    critical = critical or critical_sizes(s_co, params)
    upper = params.max_size - 1
    exceeded = {name: value for name, value in critical.items() if value > upper}
    if exceeded:
        critical_desc = ", ".join(f"{name}={value}" for name, value in sorted(critical.items()))
        exceeded_desc = ", ".join(f"{name}={value}" for name, value in sorted(exceeded.items()))
        raise ValueError(
            "Stable reconstruction critical sizes exceed the represented state domain "
            f"for max_size={params.max_size} and s_co={s_co}: {critical_desc}; "
            f"represented states stop at {upper}. Out-of-domain sizes: {exceeded_desc}."
        )
    return critical


def _channel_area(size: int, closure: ChannelClosure) -> float:
    return float(size) ** closure.area_exponent


def _activated_prefactor(closure: ChannelClosure) -> float:
    return closure.prefactor_s * math.exp(-closure.activation_barrier_over_kbt)


def _forward_frequency_from_channel(size: int, driving: float, closure: ChannelClosure) -> float:
    area = _channel_area(size, closure)
    prefactor = _activated_prefactor(closure)
    if closure.family == "legacy_water":
        return prefactor * math.exp(closure.driving_exponential * driving) * area
    if closure.family == "activated_interfacial":
        return prefactor * area
    raise ValueError(f"Unsupported channel family: {closure.family}")


def delta_w_f(
    i: int,
    n: int,
    s_co: float,
    params: Paper2022Params,
) -> float:
    return w_total(i + 1, n, s_co, params) - w_total(i, n, s_co, params)


def delta_w_g(
    n: int,
    s_co: float,
    params: Paper2022Params,
) -> float:
    del s_co
    return -params.s_cm + params.gamma_cm * ((n + 1) ** (2.0 / 3.0) - n ** (2.0 / 3.0))


def delta_w_k(
    i: int,
    s_co: float,
    params: Paper2022Params,
) -> float:
    return w_total(i + 1, i + 1, s_co, params) - w_total(i, i, s_co, params)


def forward_frequency_f(
    i: int,
    n: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.f_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(i, closure) * math.exp(
            -0.5 * delta_w_f(i, n, s_co, params)
        )
    base = _forward_frequency_from_channel(i, s_co, closure)
    if closure.family == "legacy_water" and n == i:
        return closure.split_fraction * base
    return base


def forward_frequency_g(
    n: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.g_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(n, closure) * math.exp(
            -0.5 * delta_w_g(n, s_co, params)
        )
    driving_term = params.s_cm if closure.family == "legacy_water" else s_co
    return _forward_frequency_from_channel(n, driving_term, closure)


def forward_frequency_k(
    i: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.k_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(i, closure) * math.exp(
            -0.5 * delta_w_k(i, s_co, params)
        )
    base = _forward_frequency_from_channel(i, s_co, closure)
    if closure.family == "legacy_water":
        return closure.split_fraction * base
    return base


def reverse_frequency_f(
    i: int,
    n: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    if i <= 1:
        return 0.0
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.f_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(i - 1, closure) * math.exp(
            +0.5 * delta_w_f(i - 1, n, s_co, params)
        )
    prev_forward = forward_frequency_f(i - 1, n, s_co, params, closures)
    return prev_forward * math.exp(w_total(i, n, s_co, params) - w_total(i - 1, n, s_co, params))


def reverse_frequency_g(
    n: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    if n <= 1:
        return 0.0
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.g_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(n - 1, closure) * math.exp(
            +0.5 * delta_w_g(n - 1, s_co, params)
        )
    prev_forward = forward_frequency_g(n - 1, s_co, params, closures)
    return prev_forward * math.exp(delta_w_g(n - 1, s_co, params))


def reverse_frequency_k(
    i: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    if i <= 1:
        return 0.0
    closures = _resolve_closure_set(params, closure_set)
    closure = closures.k_channel
    if closure.family == "turnbull_fisher":
        return _activated_prefactor(closure) * _channel_area(i - 1, closure) * math.exp(
            +0.5 * delta_w_k(i - 1, s_co, params)
        )
    prev_forward = forward_frequency_k(i - 1, s_co, params, closures)
    return prev_forward * math.exp(w_total(i, i, s_co, params) - w_total(i - 1, i - 1, s_co, params))


def f_attach(
    i: int,
    n: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    return forward_frequency_f(i, n, s_co, params, closure_set)


def g_attach(
    n: int,
    params: Paper2022Params,
    s_co: float | None = None,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    closures = _resolve_closure_set(params, closure_set)
    if s_co is None:
        s_co = params.s_cm
    return forward_frequency_g(n, s_co, params, closures)


def k_attach(
    i: int,
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    return forward_frequency_k(i, s_co, params, closure_set)


def build_coefficients(
    w: np.ndarray,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, np.ndarray]:
    closures = _resolve_closure_set(params, closure_set)
    rate_ref = _resolve_reference_rate(params, closures)
    a = np.zeros(len(state_pairs), dtype=float)
    b = np.zeros(len(state_pairs), dtype=float)
    c = np.zeros(len(state_pairs), dtype=float)
    d = np.zeros(len(state_pairs), dtype=float)
    e = np.zeros(len(state_pairs), dtype=float)
    h = np.zeros(len(state_pairs), dtype=float)

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)

        b[idx] = f_attach(i, n, s_co, params, closure_set=closures) / rate_ref

        if n >= 2:
            c[idx] = reverse_frequency_g(n, s_co, params, closure_set=closures) / rate_ref

        if n < i:
            d[idx] = g_attach(n, params, s_co=s_co, closure_set=closures) / rate_ref

            if i >= 2 and n <= i - 1:
                a[idx] = reverse_frequency_f(i, n, s_co, params, closure_set=closures) / rate_ref

        if n == i:
            h[idx] = k_attach(i, s_co, params, closure_set=closures) / rate_ref
            if i >= 2:
                e[idx] = reverse_frequency_k(i, s_co, params, closure_set=closures) / rate_ref

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "h": h}


def jacobian_sparsity_pattern(
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
):
    if sp is None:
        raise ImportError("scipy is required for jacobian_sparsity_pattern")

    rows: List[int] = []
    cols: List[int] = []

    def add(row: int, col: int) -> None:
        rows.append(row)
        cols.append(col)

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)

        add(idx, idx)

        if (i - 1) >= 1 and n <= (i - 1):
            add(idx, pair_to_index[(i - 1, n)])
        if (i + 1) <= (max_size - 1):
            add(idx, pair_to_index[(i + 1, n)])
        if (n - 1) >= 1:
            add(idx, pair_to_index[(i, n - 1)])
        if (n + 1) <= i:
            add(idx, pair_to_index[(i, n + 1)])

        if n == i:
            if (i - 1) >= 1:
                add(idx, pair_to_index[(i - 1, i - 1)])
            if (i + 1) <= (max_size - 1):
                add(idx, pair_to_index[(i + 1, i + 1)])

    return sp.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(state_pairs), len(state_pairs)),
    )


def build_state_topology(
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
) -> Dict[str, np.ndarray]:
    size = len(state_pairs)
    prev_i = np.full(size, -1, dtype=int)
    next_i = np.full(size, -1, dtype=int)
    prev_n = np.full(size, -1, dtype=int)
    next_n = np.full(size, -1, dtype=int)
    prev_diag = np.full(size, -1, dtype=int)
    next_diag = np.full(size, -1, dtype=int)
    is_diag = np.zeros(size, dtype=bool)

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)

        if n < i and i >= 2:
            prev_i[idx] = pair_to_index[(i - 1, n)]
        if (i + 1) <= (max_size - 1):
            next_i[idx] = pair_to_index[(i + 1, n)]
        if n >= 2:
            prev_n[idx] = pair_to_index[(i, n - 1)]
        if n < i:
            next_n[idx] = pair_to_index[(i, n + 1)]
        if n == i:
            is_diag[idx] = True
            if i >= 2:
                prev_diag[idx] = pair_to_index[(i - 1, i - 1)]
            if (i + 1) <= (max_size - 1):
                next_diag[idx] = pair_to_index[(i + 1, i + 1)]

    return {
        "prev_i": prev_i,
        "next_i": next_i,
        "prev_n": prev_n,
        "next_n": next_n,
        "prev_diag": prev_diag,
        "next_diag": next_diag,
        "is_diag": is_diag,
    }


def rhs_dimensionless_reference(
    _x: float,
    f_flat: np.ndarray,
    coeffs: Dict[str, np.ndarray],
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
) -> np.ndarray:
    a = coeffs["a"]
    b = coeffs["b"]
    c = coeffs["c"]
    d = coeffs["d"]
    e = coeffs["e"]
    h = coeffs["h"]

    values = f_flat.copy()
    values[pair_to_index[(1, 1)]] = 1.0
    deriv = np.zeros_like(values)

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)
        delta = 0.0

        if n < i and i >= 2:
            delta += a[idx] * (values[pair_to_index[(i - 1, n)]] - values[idx])

        next_val = values[pair_to_index[(i + 1, n)]] if (i + 1) <= (max_size - 1) else 0.0
        delta -= b[idx] * (values[idx] - next_val)

        if n >= 2:
            delta += c[idx] * (values[pair_to_index[(i, n - 1)]] - values[idx])

        if n < i:
            next_val = values[pair_to_index[(i, n + 1)]]
            delta -= d[idx] * (values[idx] - next_val)

        if n == i:
            if i >= 2:
                delta += e[idx] * (values[pair_to_index[(i - 1, i - 1)]] - values[idx])
            next_val = values[pair_to_index[(i + 1, i + 1)]] if (i + 1) <= (max_size - 1) else 0.0
            delta -= h[idx] * (values[idx] - next_val)

        deriv[idx] = delta

    deriv[pair_to_index[(1, 1)]] = 0.0
    return deriv


def rhs_dimensionless(
    _x: float,
    f_flat: np.ndarray,
    coeffs: Dict[str, np.ndarray],
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
    topology: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if topology is None:
        return rhs_dimensionless_reference(_x, f_flat, coeffs, pair_to_index, state_pairs, max_size)

    a = coeffs["a"]
    b = coeffs["b"]
    c = coeffs["c"]
    d = coeffs["d"]
    e = coeffs["e"]
    h = coeffs["h"]

    values = f_flat.copy()
    values[pair_to_index[(1, 1)]] = 1.0
    deriv = np.zeros_like(values)

    prev_i = topology["prev_i"]
    next_i = topology["next_i"]
    prev_n = topology["prev_n"]
    next_n = topology["next_n"]
    prev_diag = topology["prev_diag"]
    next_diag = topology["next_diag"]
    is_diag = topology["is_diag"]

    mask = prev_i >= 0
    deriv[mask] += a[mask] * (values[prev_i[mask]] - values[mask])

    mask = next_i >= 0
    deriv[mask] -= b[mask] * (values[mask] - values[next_i[mask]])

    boundary_mask = next_i < 0
    deriv[boundary_mask] -= b[boundary_mask] * values[boundary_mask]

    mask = prev_n >= 0
    deriv[mask] += c[mask] * (values[prev_n[mask]] - values[mask])

    mask = next_n >= 0
    deriv[mask] -= d[mask] * (values[mask] - values[next_n[mask]])

    mask = prev_diag >= 0
    deriv[mask] += e[mask] * (values[prev_diag[mask]] - values[mask])

    if np.any(is_diag):
        diag_mask = is_diag
        diag_next_values = np.zeros(np.count_nonzero(diag_mask), dtype=float)
        diag_next_idx = next_diag[diag_mask]
        has_next = diag_next_idx >= 0
        diag_next_values[has_next] = values[diag_next_idx[has_next]]
        deriv[diag_mask] -= h[diag_mask] * (values[diag_mask] - diag_next_values)

    deriv[pair_to_index[(1, 1)]] = 0.0
    return deriv


def _build_stationary_linear_system(
    coeffs: Dict[str, np.ndarray],
    topology: Dict[str, np.ndarray],
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
) -> Dict[str, object]:
    if sp is None:
        raise ImportError("scipy is required for stationary linear system assembly")

    anchor_idx = pair_to_index[(1, 1)]
    size = len(state_pairs)
    unknown_indices = [idx for idx in range(size) if idx != anchor_idx]
    local_index = {idx: pos for pos, idx in enumerate(unknown_indices)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs = np.zeros(len(unknown_indices), dtype=float)

    def add_neighbor(row_idx: int, state_idx: int, coeff: float) -> None:
        if coeff == 0.0:
            return
        if state_idx == anchor_idx:
            rhs[row_idx] -= coeff
        else:
            rows.append(row_idx)
            cols.append(local_index[state_idx])
            data.append(coeff)

    prev_i = topology["prev_i"]
    next_i = topology["next_i"]
    prev_n = topology["prev_n"]
    next_n = topology["next_n"]
    prev_diag = topology["prev_diag"]
    next_diag = topology["next_diag"]

    for state_idx in unknown_indices:
        row_idx = local_index[state_idx]
        diag_coeff = 0.0

        if prev_i[state_idx] >= 0:
            coeff = float(coeffs["a"][state_idx])
            add_neighbor(row_idx, int(prev_i[state_idx]), coeff)
            diag_coeff -= coeff

        coeff = float(coeffs["b"][state_idx])
        diag_coeff -= coeff
        if next_i[state_idx] >= 0:
            add_neighbor(row_idx, int(next_i[state_idx]), coeff)

        if prev_n[state_idx] >= 0:
            coeff = float(coeffs["c"][state_idx])
            add_neighbor(row_idx, int(prev_n[state_idx]), coeff)
            diag_coeff -= coeff

        if next_n[state_idx] >= 0:
            coeff = float(coeffs["d"][state_idx])
            diag_coeff -= coeff
            add_neighbor(row_idx, int(next_n[state_idx]), coeff)

        if prev_diag[state_idx] >= 0:
            coeff = float(coeffs["e"][state_idx])
            add_neighbor(row_idx, int(prev_diag[state_idx]), coeff)
            diag_coeff -= coeff

        coeff = float(coeffs["h"][state_idx])
        diag_coeff -= coeff
        if next_diag[state_idx] >= 0:
            add_neighbor(row_idx, int(next_diag[state_idx]), coeff)

        rows.append(row_idx)
        cols.append(row_idx)
        data.append(diag_coeff)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(unknown_indices), len(unknown_indices)))
    return {
        "matrix": matrix,
        "rhs": rhs,
        "anchor_idx": anchor_idx,
        "unknown_indices": unknown_indices,
    }


def _build_transient_linear_system(
    coeffs: Dict[str, np.ndarray],
    topology: Dict[str, np.ndarray],
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
) -> Dict[str, object]:
    stationary = _build_stationary_linear_system(coeffs, topology, pair_to_index, state_pairs)
    return {
        "matrix": stationary["matrix"],
        "source": -stationary["rhs"],
        "anchor_idx": stationary["anchor_idx"],
        "unknown_indices": stationary["unknown_indices"],
    }


def _reconstruct_full_state_trajectory(
    state_count: int,
    anchor_idx: int,
    unknown_indices: List[int],
    unknown_trajectory: np.ndarray,
) -> np.ndarray:
    y = np.zeros((state_count, unknown_trajectory.shape[1]), dtype=float)
    y[anchor_idx, :] = 1.0
    for local_idx, state_idx in enumerate(unknown_indices):
        y[state_idx, :] = unknown_trajectory[local_idx, :]
    return y


def _solve_stationary_linear_system(
    matrix,
    rhs: np.ndarray,
    refinement_steps: int = 3,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if sp is None or spla is None:
        raise ImportError("scipy.sparse.linalg is required for stationary linear solve")

    row_scale = np.abs(matrix).max(axis=1).toarray().ravel()
    row_scale = np.where(row_scale > 0.0, row_scale, 1.0)
    inv_row_scale = 1.0 / row_scale

    scaled_matrix = sp.diags(inv_row_scale) @ matrix
    scaled_rhs = rhs * inv_row_scale

    lu = spla.splu(scaled_matrix.tocsc())
    current = lu.solve(scaled_rhs)

    def residual_metrics(candidate: np.ndarray) -> Tuple[float, float]:
        residual = rhs - matrix.dot(candidate)
        return float(np.linalg.norm(residual, ord=np.inf)), float(np.linalg.norm(residual))

    residual_history = []
    best = current.copy()
    best_iteration = 0
    best_metrics = residual_metrics(best)
    residual_history.append(best_metrics)

    for iteration in range(1, refinement_steps + 1):
        residual = rhs - matrix.dot(current)
        correction = lu.solve(residual * inv_row_scale)
        current = current + correction
        metrics = residual_metrics(current)
        residual_history.append(metrics)
        if metrics[0] < best_metrics[0]:
            best = current.copy()
            best_iteration = iteration
            best_metrics = metrics

    return best, {
        "row_scale_min": float(np.min(row_scale)),
        "row_scale_max": float(np.max(row_scale)),
        "refinement_steps": int(refinement_steps),
        "residual_history": residual_history,
        "selected_iteration": int(best_iteration),
        "selected_residual_inf": float(best_metrics[0]),
        "selected_residual_l2": float(best_metrics[1]),
        "best_refined_residual_inf": float(min(metric[0] for metric in residual_history)),
        "best_refined_residual_l2": float(min(metric[1] for metric in residual_history)),
    }


def solve_dimensionless_system(
    s_co: float,
    params: Paper2022Params,
    time_s: np.ndarray,
    method: str = "BDF",
    max_step_s: float = 1.0e-7,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
    closure_set: TwoStepClosureSet | None = None,
):
    if sp is None:
        raise ImportError("scipy is required for solve_dimensionless_system")

    pair_to_index, state_pairs = triangular_index_maps(params.max_size)
    w, c_eq, c1 = make_w_and_c_arrays(s_co, params, state_pairs)
    closures = _resolve_closure_set(params, closure_set)
    rate_ref = _resolve_reference_rate(params, closures)
    coeffs = build_coefficients(
        w,
        s_co,
        params,
        pair_to_index,
        state_pairs,
        closure_set=closures,
    )
    sparsity = jacobian_sparsity_pattern(pair_to_index, state_pairs, params.max_size)
    topology = build_state_topology(pair_to_index, state_pairs, params.max_size)

    x_eval = rate_ref * np.asarray(time_s, dtype=float)
    f0_state = np.zeros(len(state_pairs), dtype=float)
    f0_state[pair_to_index[(1, 1)]] = 1.0
    stationary_diagnostics = None

    if method == "stationary":
        if sp is None or spla is None:
            raise ImportError("scipy.sparse.linalg is required for method='stationary'")

        system = _build_stationary_linear_system(coeffs, topology, pair_to_index, state_pairs)
        plain_unknown = spla.spsolve(system["matrix"], system["rhs"])
        scaled_unknown, stationary_diagnostics = _solve_stationary_linear_system(
            system["matrix"],
            system["rhs"],
            refinement_steps=3,
        )

        anchor_idx = int(system["anchor_idx"])
        plain_state = np.zeros(len(state_pairs), dtype=float)
        plain_state[anchor_idx] = 1.0
        for idx, value in zip(system["unknown_indices"], plain_unknown):
            plain_state[idx] = float(value)

        scaled_state = np.zeros(len(state_pairs), dtype=float)
        scaled_state[anchor_idx] = 1.0
        for idx, value in zip(system["unknown_indices"], scaled_unknown):
            scaled_state[idx] = float(value)

        plain_rates = rate_dict_from_state_stable(
            plain_state,
            w,
            c1,
            s_co,
            params,
            pair_to_index,
            closure_set=closures,
        )
        scaled_rates = rate_dict_from_state_stable(
            scaled_state,
            w,
            c1,
            s_co,
            params,
            pair_to_index,
            closure_set=closures,
        )

        plain_residual = system["rhs"] - system["matrix"].dot(plain_unknown)
        scaled_residual = system["rhs"] - system["matrix"].dot(scaled_unknown)
        plain_residual_inf = float(np.linalg.norm(plain_residual, ord=np.inf))
        scaled_residual_inf = float(np.linalg.norm(scaled_residual, ord=np.inf))
        plain_residual_l2 = float(np.linalg.norm(plain_residual))
        scaled_residual_l2 = float(np.linalg.norm(scaled_residual))

        primary_rate_names = ("J_d", "J_d_plus_com", "J_com", "J_tot")

        def has_valid_primary_rates(rates: Dict[str, float]) -> bool:
            return all(
                np.isfinite(rates[name]) and rates[name] > 0.0
                for name in primary_rate_names
            )

        def has_valid_grouped_rate(rates: Dict[str, float]) -> bool:
            return bool(np.isfinite(rates["J_c"]) and rates["J_c"] > 0.0)

        candidates = [
            {
                "path": "plain",
                "state": plain_state,
                "rates": plain_rates,
                "primary_valid": has_valid_primary_rates(plain_rates),
                "grouped_valid": has_valid_grouped_rate(plain_rates),
                "residual_inf": plain_residual_inf,
                "residual_l2": plain_residual_l2,
            },
            {
                "path": "row_scaled_refined",
                "state": scaled_state,
                "rates": scaled_rates,
                "primary_valid": has_valid_primary_rates(scaled_rates),
                "grouped_valid": has_valid_grouped_rate(scaled_rates),
                "residual_inf": scaled_residual_inf,
                "residual_l2": scaled_residual_l2,
            },
        ]
        for candidate in candidates:
            candidate["valid"] = bool(candidate["primary_valid"] and candidate["grouped_valid"])
        valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
        primary_valid_candidates = [candidate for candidate in candidates if candidate["primary_valid"]]
        selection_pool = valid_candidates or primary_valid_candidates or candidates
        selected_candidate = min(selection_pool, key=lambda candidate: candidate["residual_inf"])
        stationary_state = selected_candidate["state"]
        stationary_diagnostics["selected_path"] = str(selected_candidate["path"])
        stationary_diagnostics["plain_J_c"] = float(plain_rates["J_c"])
        stationary_diagnostics["scaled_J_c"] = float(scaled_rates["J_c"])
        for name in primary_rate_names:
            stationary_diagnostics[f"plain_{name}"] = float(plain_rates[name])
            stationary_diagnostics[f"scaled_{name}"] = float(scaled_rates[name])
        stationary_diagnostics["plain_primary_valid"] = bool(has_valid_primary_rates(plain_rates))
        stationary_diagnostics["scaled_primary_valid"] = bool(has_valid_primary_rates(scaled_rates))
        stationary_diagnostics["plain_grouped_valid"] = bool(has_valid_grouped_rate(plain_rates))
        stationary_diagnostics["scaled_grouped_valid"] = bool(has_valid_grouped_rate(scaled_rates))
        stationary_diagnostics["plain_residual_inf"] = plain_residual_inf
        stationary_diagnostics["scaled_residual_inf"] = scaled_residual_inf
        stationary_diagnostics["plain_residual_l2"] = plain_residual_l2
        stationary_diagnostics["scaled_residual_l2"] = scaled_residual_l2

        solution = SimpleNamespace(
            y=np.repeat(stationary_state[:, None], len(x_eval), axis=1),
            t=x_eval,
            success=True,
            message="stationary sparse solve",
        )
    elif method == "expm":
        if sp is None or spla is None:
            raise ImportError("scipy.sparse.linalg is required for method='expm'")

        system = _build_transient_linear_system(coeffs, topology, pair_to_index, state_pairs)
        operator = system["matrix"].tocsr()
        unknown_indices = system["unknown_indices"]
        anchor_idx = int(system["anchor_idx"])
        rhs = -np.asarray(system["source"], dtype=float)
        stationary_unknown, _ = _solve_stationary_linear_system(
            operator,
            rhs,
            refinement_steps=3,
        )
        delta0 = -stationary_unknown

        y = np.zeros((len(state_pairs), len(x_eval)), dtype=float)
        y[anchor_idx, :] = 1.0
        for col_idx, x_value in enumerate(x_eval):
            if x_value == 0.0:
                unknown = np.zeros_like(stationary_unknown)
            else:
                homogeneous = spla.expm_multiply(operator * float(x_value), delta0)
                unknown = stationary_unknown + homogeneous
            for local_idx, state_idx in enumerate(unknown_indices):
                y[state_idx, col_idx] = float(unknown[local_idx])

        solution = SimpleNamespace(
            y=y,
            t=x_eval,
            success=True,
            message="matrix exponential solve",
        )
    elif method.endswith("_linear"):
        if solve_ivp is None:
            raise ImportError("scipy.integrate.solve_ivp is required for linear transient solve")

        base_method = method[: -len("_linear")]
        if base_method not in {"BDF", "Radau", "LSODA"}:
            raise ValueError(f"unsupported linear transient method: {method}")

        system = _build_transient_linear_system(coeffs, topology, pair_to_index, state_pairs)
        operator = system["matrix"].tocsr()
        source = np.asarray(system["source"], dtype=float)
        unknown_indices = system["unknown_indices"]
        anchor_idx = int(system["anchor_idx"])
        u0 = np.zeros(len(unknown_indices), dtype=float)

        solve_kwargs = {
            "fun": lambda x, u: operator.dot(u) + source,
            "t_span": (0.0, float(x_eval[-1])),
            "y0": u0,
            "method": base_method,
            "t_eval": x_eval,
            "rtol": rtol,
            "atol": atol,
        }
        if base_method in {"BDF", "Radau"}:
            solve_kwargs["jac"] = operator
        if max_step_s is not None:
            solve_kwargs["max_step"] = rate_ref * max_step_s

        reduced_solution = solve_ivp(**solve_kwargs)
        if not reduced_solution.success:
            raise RuntimeError(reduced_solution.message)

        y = _reconstruct_full_state_trajectory(
            len(state_pairs),
            anchor_idx,
            unknown_indices,
            np.asarray(reduced_solution.y, dtype=float),
        )
        solution = SimpleNamespace(
            y=y,
            t=x_eval,
            success=True,
            message=f"{base_method} linear reduced solve",
        )
    elif method.endswith("_linear_c"):
        # c-variable BDF: integrate dc/dt = M*c + s_c in actual concentration space
        # rather than f = c/c_eq dimensionless space. This avoids the post-critical
        # noise amplification that plagues the standard _linear method, because BDF
        # tracks the small actual concentration values (with rtol*c absolute precision)
        # instead of f-values that stay near unity in regions where true c is ~1e-50.
        if solve_ivp is None:
            raise ImportError("scipy.integrate.solve_ivp is required for linear transient solve")

        base_method = method[: -len("_linear_c")]
        if base_method not in {"BDF", "Radau", "LSODA"}:
            raise ValueError(f"unsupported c-variable linear transient method: {method}")

        system = _build_transient_linear_system(coeffs, topology, pair_to_index, state_pairs)
        A_f = system["matrix"].tocsr()
        s_f = np.asarray(system["source"], dtype=float)
        unknown_indices = system["unknown_indices"]
        anchor_idx = int(system["anchor_idx"])
        c_eq_unknown = c_eq[unknown_indices]

        # Conjugate the operator: M = diag(c_eq) @ A @ diag(1/c_eq)
        # Element-wise: M[i,j] = (c_eq[i] / c_eq[j]) * A[i,j]
        A_coo = A_f.tocoo()
        log_c_eq_u = np.log(np.maximum(c_eq_unknown, 1e-300))
        log_ratio = log_c_eq_u[A_coo.row] - log_c_eq_u[A_coo.col]
        log_ratio_clipped = np.clip(log_ratio, -700, 700)
        new_data = A_coo.data * np.exp(log_ratio_clipped)
        M = sp.csr_matrix((new_data, (A_coo.row, A_coo.col)), shape=A_f.shape)
        # Source in c-space: s_c = c_eq * s_f
        s_c = c_eq_unknown * s_f

        # IC: only monomer present, all other concentrations = 0
        c0_unknown = np.zeros(len(unknown_indices), dtype=float)

        # Use very small atol (1e-30) because c values are physically small in deep states
        solve_kwargs = {
            "fun": lambda x, c: M.dot(c) + s_c,
            "t_span": (0.0, float(x_eval[-1])),
            "y0": c0_unknown,
            "method": base_method,
            "t_eval": x_eval,
            "rtol": rtol,
            "atol": 1.0e-30,
        }
        if base_method in {"BDF", "Radau"}:
            solve_kwargs["jac"] = M
        if max_step_s is not None:
            solve_kwargs["max_step"] = rate_ref * max_step_s

        reduced_solution = solve_ivp(**solve_kwargs)
        if not reduced_solution.success:
            raise RuntimeError(reduced_solution.message)

        # Convert back to f-space (existing rate computation expects f)
        c_traj = np.asarray(reduced_solution.y, dtype=float)
        f_traj = c_traj / c_eq_unknown[:, None]
        y = _reconstruct_full_state_trajectory(
            len(state_pairs), anchor_idx, unknown_indices, f_traj,
        )
        solution = SimpleNamespace(
            y=y,
            t=x_eval,
            success=True,
            message=f"{base_method} c-variable linear reduced solve",
        )
    elif method == "odeint":
        if odeint is None:
            raise ImportError("scipy.integrate.odeint is required for method='odeint'")

        odeint_kwargs = {
            "func": lambda y, x: rhs_dimensionless(
                x, y, coeffs, pair_to_index, state_pairs, params.max_size, topology=topology
            ),
            "y0": f0_state,
            "t": x_eval,
            "rtol": rtol,
            "atol": atol,
            "full_output": True,
        }
        if max_step_s is not None:
            odeint_kwargs["hmax"] = rate_ref * max_step_s

        y_t, info = odeint(**odeint_kwargs)
        message = info.get("message", "")
        if "successful" not in message.lower():
            raise RuntimeError(message)
        y_t[:, pair_to_index[(1, 1)]] = 1.0
        solution = SimpleNamespace(
            y=y_t.T,
            t=x_eval,
            success=True,
            message=message,
        )
    else:
        if solve_ivp is None:
            raise ImportError("scipy.integrate.solve_ivp is required for solve_dimensionless_system")

        solve_kwargs = {
            "fun": lambda x, y: rhs_dimensionless(
                x, y, coeffs, pair_to_index, state_pairs, params.max_size, topology=topology
            ),
            "t_span": (0.0, float(x_eval[-1])),
            "y0": f0_state,
            "method": method,
            "t_eval": x_eval,
            "rtol": rtol,
            "atol": atol,
        }
        if method in {"BDF", "Radau"}:
            solve_kwargs["jac_sparsity"] = sparsity
        if max_step_s is not None:
            solve_kwargs["max_step"] = rate_ref * max_step_s

        solution = solve_ivp(**solve_kwargs)
        if not solution.success:
            raise RuntimeError(solution.message)
        solution.y[pair_to_index[(1, 1)], :] = 1.0

    return {
        "solution": solution,
        "pair_to_index": pair_to_index,
        "state_pairs": state_pairs,
        "w": w,
        "c_eq": c_eq,
        "c1": c1,
        "coeffs": coeffs,
        "closure_set": closures,
        "topology": topology,
        "stationary_diagnostics": stationary_diagnostics,
        "time_s": np.asarray(time_s, dtype=float),
        "x_eval": x_eval,
        "time_reference_s_inv": rate_ref,
    }


def solve_one_step_nucleation_system(
    s_co: float,
    params: Paper2022Params,
    time_s: np.ndarray,
    mode: str,
    method: str = "BDF",
    max_step_s: float = 1.0e-7,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
):
    if solve_ivp is None:
        raise ImportError("scipy.integrate.solve_ivp is required for solve_one_step_nucleation_system")

    if mode not in {"crystal", "droplet"}:
        raise ValueError(f"unsupported 1S mode: {mode}")

    sizes = np.arange(1, params.max_size, dtype=int)
    x_eval = params.f0 * np.asarray(time_s, dtype=float)

    if mode == "crystal":
        w_line = np.array([w_total(int(i), int(i), s_co, params) for i in sizes], dtype=float)
        threshold_size = critical_sizes(s_co, params)["i_co_star"]
    else:
        w_line = np.array([w_total(int(i), 1, s_co, params) for i in sizes], dtype=float)
        threshold_size = critical_sizes(s_co, params)["i_star"]

    c1 = params.c1_base * np.exp(s_co)
    w11 = w_total(1, 1, s_co, params)
    c_eq = _exp_from_log_clipped(math.log(float(c1)) + w11 - w_line)

    # In the 1S limits discussed in the paper, the growth frequency along the
    # active one-dimensional pathway is the CNT direct-impingement form.
    forward = np.exp(s_co) * sizes.astype(float) ** (2.0 / 3.0)
    backward = np.zeros_like(forward)
    backward[1:] = forward[:-1] * np.exp(w_line[1:] - w_line[:-1])

    def rhs_line(_x: float, f_line: np.ndarray) -> np.ndarray:
        values = f_line.copy()
        values[0] = 1.0

        deriv = np.zeros_like(values)
        current = values[1:]
        previous = values[:-1]
        next_values = np.zeros_like(current)
        if len(values) > 2:
            next_values[:-1] = values[2:]

        deriv[1:] = backward[1:] * (previous - current) - forward[1:] * (current - next_values)
        return deriv

    y0 = np.zeros(len(sizes), dtype=float)
    y0[0] = 1.0

    solve_kwargs = {
        "fun": rhs_line,
        "t_span": (0.0, float(x_eval[-1])),
        "y0": y0,
        "method": method,
        "t_eval": x_eval,
        "rtol": rtol,
        "atol": atol,
    }
    if sp is not None and method in {"BDF", "Radau"}:
        solve_kwargs["jac_sparsity"] = sp.diags(
            diagonals=[np.ones(len(sizes) - 1), np.ones(len(sizes)), np.ones(len(sizes) - 1)],
            offsets=[-1, 0, 1],
            format="csr",
        )
    if max_step_s is not None:
        solve_kwargs["max_step"] = params.f0 * max_step_s

    solution = solve_ivp(**solve_kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    solution.y[0, :] = 1.0

    return {
        "solution": solution,
        "sizes": sizes,
        "w_line": w_line,
        "c_eq": c_eq,
        "c1": c1,
        "forward": params.f0 * forward,
        "mode": mode,
        "threshold_size": threshold_size,
        "time_s": np.asarray(time_s, dtype=float),
        "x_eval": x_eval,
    }


def one_step_rate_time_series(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
    mode: str,
) -> np.ndarray:
    if mode not in {"crystal", "droplet"}:
        raise ValueError(f"unsupported 1S mode: {mode}")

    values = np.array(solve_output["solution"].y, dtype=float)
    values[0, :] = 1.0
    next_values = np.zeros_like(values)
    next_values[:-1, :] = values[1:, :]

    flux = solve_output["forward"][:, None] * solve_output["c_eq"][:, None] * (values - next_values)
    threshold_idx = int(solve_output["threshold_size"]) - 1
    return flux[threshold_idx, :]


def _logsumexp(log_values: List[float]) -> float:
    if not log_values:
        return float("-inf")
    max_log = max(log_values)
    if not math.isfinite(max_log):
        return max_log
    return max_log + math.log(sum(math.exp(value - max_log) for value in log_values))


def _signed_logsumexp(terms: List[Tuple[int, float]]) -> Tuple[int, float]:
    finite_terms = [(sign, logabs) for sign, logabs in terms if sign != 0 and math.isfinite(logabs)]
    if not finite_terms:
        return 0, float("-inf")

    max_log = max(logabs for _, logabs in finite_terms)
    scaled_sum = math.fsum(sign * math.exp(logabs - max_log) for sign, logabs in finite_terms)
    if scaled_sum == 0.0:
        return 0, float("-inf")
    return (1 if scaled_sum > 0.0 else -1), max_log + math.log(abs(scaled_sum))


def _signed_log_to_float(sign: int, logabs: float) -> float:
    if sign == 0 or not math.isfinite(logabs):
        return 0.0
    if logabs >= _LOG_FLOAT_MAX:
        return (1 if sign > 0 else -1) * _FLOAT_MAX
    return sign * math.exp(logabs)


def _signed_logsumexp_diagnostics(terms: List[Tuple[int, float]]) -> Dict[str, float]:
    finite_terms = [(sign, logabs) for sign, logabs in terms if sign != 0 and math.isfinite(logabs)]
    sign, logabs = _signed_logsumexp(finite_terms)
    abs_logsum = _logsumexp([logabs_term for _, logabs_term in finite_terms])
    positive_logsum = _logsumexp([logabs_term for sign_term, logabs_term in finite_terms if sign_term > 0])
    negative_logsum = _logsumexp([logabs_term for sign_term, logabs_term in finite_terms if sign_term < 0])

    if sign == 0 or not math.isfinite(logabs) or not math.isfinite(abs_logsum):
        cancellation_ratio = 0.0 if finite_terms else 1.0
    else:
        cancellation_ratio = math.exp(min(0.0, logabs - abs_logsum))

    return {
        "sign": float(sign),
        "logabs": float(logabs),
        "value": float(_signed_log_to_float(sign, logabs)),
        "abs_logsum": float(abs_logsum),
        "positive_logsum": float(positive_logsum),
        "negative_logsum": float(negative_logsum),
        "cancellation_ratio": float(cancellation_ratio),
        "term_count": float(len(finite_terms)),
    }


def _exp_from_log_clipped(log_values):
    log_array = np.asarray(log_values, dtype=float)
    out = np.zeros_like(log_array)

    high_mask = log_array >= _LOG_FLOAT_MAX
    low_mask = log_array <= _LOG_FLOAT_MIN
    mid_mask = ~(high_mask | low_mask)

    out[high_mask] = _FLOAT_MAX
    out[mid_mask] = np.exp(log_array[mid_mask])
    out[low_mask] = 0.0

    if np.isscalar(log_values):
        return float(out)
    return out


def _signed_log_to_float_clipped(sign: int, logabs: float, clip_logabs: float = _LOG_FLUX_CLIP) -> float:
    if sign == 0 or not math.isfinite(logabs):
        return 0.0
    return sign * math.exp(min(logabs, clip_logabs))


def _signed_log_flux_term(log_prefactor: float, current_value: float, next_value: float) -> Tuple[int, float]:
    difference = float(current_value - next_value)
    if difference == 0.0:
        return 0, float("-inf")
    return (1 if difference > 0.0 else -1), log_prefactor + math.log(abs(difference))


def _closure_set_from_solve_output(
    solve_output: Dict[str, object],
    closure_set: TwoStepClosureSet | None,
) -> TwoStepClosureSet | None:
    stored = solve_output.get("closure_set")
    if stored is None:
        return closure_set
    if closure_set is None:
        return cast(TwoStepClosureSet, stored)
    if closure_set != stored:
        raise ValueError("explicit closure_set conflicts with solve_output['closure_set']")
    return cast(TwoStepClosureSet, stored)


def decompose_j_c_terms_stable(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, object]:
    closures = _resolve_closure_set(params, closure_set)
    critical = _validate_stable_reconstruction_domain(s_co, params)
    i_co_star = critical["i_co_star"]
    upper = params.max_size

    log_c1 = math.log(float(c1))
    w11 = float(w[pair_to_index[(1, 1)]])
    log_c_eq = log_c1 + w11 - w

    def next_i_value(i: int, n: int) -> float:
        if i <= (upper - 2):
            return float(f_flat[pair_to_index[(i + 1, n)]])
        return 0.0

    def next_diagonal_value(i: int) -> float:
        if i <= (upper - 2):
            return float(f_flat[pair_to_index[(i + 1, i + 1)]])
        return 0.0

    def make_term(kind: str, i: int, n: int | None, sign: int, logabs: float) -> Dict[str, object]:
        return {
            "kind": kind,
            "i": int(i),
            "n": None if n is None else int(n),
            "sign": int(sign),
            "logabs": float(logabs),
            "value": _signed_log_to_float(sign, logabs),
        }

    def i_flux_term(i: int, n: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, n)]
        log_pref = math.log(f_attach(i, n, s_co, params, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), next_i_value(i, n))

    def g_flux_term(i: int, n: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, n)]
        log_pref = math.log(g_attach(n, params, s_co=s_co, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), float(f_flat[pair_to_index[(i, n + 1)]]))

    def k_flux_term(i: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, i)]
        log_pref = math.log(k_attach(i, s_co, params, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), next_diagonal_value(i))

    k_sign, k_logabs = k_flux_term(i_co_star)
    k_term = make_term("K", i_co_star, i_co_star, k_sign, k_logabs)

    g_terms = []
    i_terms = []
    pair_terms = []
    for i in range(i_co_star + 1, upper):
        g_sign, g_logabs = g_flux_term(i, i - 1)
        i_sign, i_logabs = i_flux_term(i, i)
        g_terms.append(make_term("G", i, i - 1, g_sign, g_logabs))
        i_terms.append(make_term("I", i, i, i_sign, i_logabs))
        pair_sign, pair_logabs = _signed_logsumexp([(g_sign, g_logabs), (-i_sign, i_logabs)])
        pair_terms.append(make_term("G_minus_I", i, i, pair_sign, pair_logabs))

    k_terms = [k_term]
    k_sum = _signed_logsumexp([(term["sign"], term["logabs"]) for term in k_terms])
    g_sum = _signed_logsumexp([(term["sign"], term["logabs"]) for term in g_terms])
    i_sum = _signed_logsumexp([(term["sign"], term["logabs"]) for term in i_terms])
    j_c_sum = _signed_logsumexp(
        [(term["sign"], term["logabs"]) for term in k_terms + pair_terms]
    )

    return {
        "K_terms": k_terms,
        "G_terms": g_terms,
        "I_terms": i_terms,
        "pair_terms": pair_terms,
        "K_sign": k_sum[0],
        "K_logabs": k_sum[1],
        "K": _signed_log_to_float(*k_sum),
        "G_sign": g_sum[0],
        "G_logabs": g_sum[1],
        "G": _signed_log_to_float(*g_sum),
        "I_sign": i_sum[0],
        "I_logabs": i_sum[1],
        "I": _signed_log_to_float(*i_sum),
        "J_c_sign": j_c_sum[0],
        "J_c_logabs": j_c_sum[1],
        "J_c": _signed_log_to_float(*j_c_sum),
    }


def _rate_terms_from_state_stable(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, List[Tuple[int, float]]]:
    closures = _resolve_closure_set(params, closure_set)
    critical = _validate_stable_reconstruction_domain(s_co, params)
    i_star = critical["i_star"]
    n_star = critical["n_star"]
    i_co_star = critical["i_co_star"]
    upper = params.max_size

    log_c1 = math.log(float(c1))
    w11 = float(w[pair_to_index[(1, 1)]])
    log_c_eq = log_c1 + w11 - w

    def next_value(i: int, n: int) -> float:
        if i <= (upper - 2):
            return float(f_flat[pair_to_index[(i + 1, n)]])
        return 0.0

    def next_crystal_value(i: int) -> float:
        if i <= (upper - 2):
            return float(f_flat[pair_to_index[(i + 1, i + 1)]])
        return 0.0

    def i_flux_term(i: int, n: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, n)]
        log_pref = math.log(f_attach(i, n, s_co, params, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), next_value(i, n))

    def g_flux_term(i: int, n: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, n)]
        log_pref = math.log(g_attach(n, params, s_co=s_co, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), float(f_flat[pair_to_index[(i, n + 1)]]))

    def k_flux_term(i: int) -> Tuple[int, float]:
        idx = pair_to_index[(i, i)]
        log_pref = math.log(k_attach(i, s_co, params, closure_set=closures)) + float(log_c_eq[idx])
        return _signed_log_flux_term(log_pref, float(f_flat[idx]), next_crystal_value(i))

    j_c_terms = [k_flux_term(i_co_star)]
    for i in range(i_co_star + 1, upper):
        g_term = g_flux_term(i, i - 1)
        i_term = i_flux_term(i, i)
        j_c_terms.append(_signed_logsumexp([g_term, (-i_term[0], i_term[1])]))

    j_c_d_terms = [i_flux_term(i, i) for i in range(n_star + 1, upper)]
    j_c_d_terms += [g_flux_term(i, n_star) for i in range(n_star + 1, upper)]
    j_c_d_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, i - 1) for i in range(n_star + 1, upper))]

    j_d_terms = [i_flux_term(i_star, 1)]
    j_d_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, 1) for i in range(i_star + 1, upper))]

    j_d_plus_com_terms = [i_flux_term(i_star, n) for n in range(1, i_star + 1)]
    j_d_plus_com_terms += [i_flux_term(i, i) for i in range(i_star + 1, upper)]
    j_d_plus_com_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, i - 1) for i in range(i_star + 1, upper))]

    j_com_terms = [i_flux_term(i_star, n) for n in range(2, i_star + 1)]
    j_com_terms += [i_flux_term(i, i) for i in range(i_star + 1, upper)]
    j_com_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, i - 1) for i in range(i_star + 1, upper))]

    j_tot_terms = [k_flux_term(i_co_star)]
    j_tot_terms += [i_flux_term(i_co_star, n) for n in range(1, i_co_star + 1)]

    j_comp_terms = [i_flux_term(i_co_star, n) for n in range(1, i_co_star + 1)]
    j_comp_terms += [i_flux_term(i, i) for i in range(i_co_star + 1, upper)]
    j_comp_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, i - 1) for i in range(i_co_star + 1, upper))]

    j_cr_met_terms = [i_flux_term(i, i) for i in range(n_star + 1, upper)]
    j_cr_met_terms += [g_flux_term(i, n_star) for i in range(n_star + 1, upper)]
    j_cr_met_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, i - 1) for i in range(n_star + 1, upper))]

    j_met_terms = [i_flux_term(i_star, 1)]
    j_met_terms += [(-sign, logabs) for sign, logabs in (g_flux_term(i, 1) for i in range(i_star + 1, upper))]

    return {
        "J_c": j_c_terms,
        "J_c_d": j_c_d_terms,
        "J_d": j_d_terms,
        "J_d_plus_com": j_d_plus_com_terms,
        "J_com": j_com_terms,
        "J_tot": j_tot_terms,
        "J_comp": j_comp_terms,
        "J_cr_met": j_cr_met_terms,
        "J_met": j_met_terms,
    }


def rate_dict_from_state_signed_stable(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, Dict[str, float]]:
    return {
        name: _signed_logsumexp_diagnostics(terms)
        for name, terms in _rate_terms_from_state_stable(
            f_flat,
            w,
            c1,
            s_co,
            params,
            pair_to_index,
            closure_set=closure_set,
        ).items()
    }


def rate_dict_from_state_stable(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, float]:
    diagnostics = rate_dict_from_state_signed_stable(
        f_flat,
        w,
        c1,
        s_co,
        params,
        pair_to_index,
        closure_set=closure_set,
    )
    return {name: float(info["value"]) for name, info in diagnostics.items()}


def final_rates_from_solve_output(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> Dict[str, float]:
    f_flat = solve_output["solution"].y[:, -1]
    closures = _closure_set_from_solve_output(solve_output, closure_set)
    return rate_dict_from_state_stable(
        f_flat,
        solve_output["w"],
        solve_output["c1"],
        s_co,
        params,
        solve_output["pair_to_index"],
        closure_set=closures,
    )


def grouped_jc_from_state(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    return decompose_j_c_terms_stable(
        f_flat,
        w,
        c1,
        s_co,
        params,
        pair_to_index,
        closure_set=closure_set,
    )["J_c"]


def grouped_jc_from_solve_output(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
    closure_set: TwoStepClosureSet | None = None,
) -> float:
    closures = _closure_set_from_solve_output(solve_output, closure_set)
    return grouped_jc_from_state(
        solve_output["solution"].y[:, -1],
        solve_output["w"],
        solve_output["c1"],
        s_co,
        params,
        solve_output["pair_to_index"],
        closure_set=closures,
    )


def restore_actual_concentration(f_flat: np.ndarray, w: np.ndarray, c1: float, s_co: float, params: Paper2022Params) -> np.ndarray:
    delta_w = w - w_total(1, 1, s_co, params)
    log_c = math.log(float(c1)) - delta_w
    return _exp_from_log_clipped(log_c) * f_flat


def log_actual_concentration(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
) -> np.ndarray:
    """Return log(Z) without forming C_eq * F in ordinary floating point first."""
    delta_w = w - w_total(1, 1, s_co, params)
    log_c_eq = math.log(float(c1)) - delta_w
    f_array = np.asarray(f_flat, dtype=float)
    out = np.full_like(f_array, -np.inf, dtype=float)
    positive = f_array > 0.0
    out[positive] = log_c_eq[positive] + np.log(f_array[positive])
    return out


def log10_actual_concentration(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
) -> np.ndarray:
    return log_actual_concentration(f_flat, w, c1, s_co, params) / math.log(10.0)


def compute_flux_maps(
    f_flat: np.ndarray,
    c_eq: np.ndarray,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
    closure_set: TwoStepClosureSet | None = None,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    closures = _resolve_closure_set(params, closure_set)
    i_flux: Dict[Tuple[int, int], float] = {}
    g_flux: Dict[Tuple[int, int], float] = {}
    k_flux: Dict[Tuple[int, int], float] = {}

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)

        next_val = f_flat[pair_to_index[(i + 1, n)]] if (i + 1) <= (max_size - 1) else 0.0
        i_flux[(i, n)] = (
            f_attach(i, n, s_co, params, closure_set=closures) * c_eq[idx] * (f_flat[idx] - next_val)
        )

        if n < i:
            next_val = f_flat[pair_to_index[(i, n + 1)]]
            g_flux[(i, n)] = (
                g_attach(n, params, s_co=s_co, closure_set=closures) * c_eq[idx] * (f_flat[idx] - next_val)
            )

        if n == i:
            next_val = f_flat[pair_to_index[(i + 1, i + 1)]] if (i + 1) <= (max_size - 1) else 0.0
            k_flux[(i, i)] = (
                k_attach(i, s_co, params, closure_set=closures) * c_eq[idx] * (f_flat[idx] - next_val)
            )

    return i_flux, g_flux, k_flux


def compute_flux_maps_stable(
    f_flat: np.ndarray,
    w: np.ndarray,
    c1: float,
    s_co: float,
    params: Paper2022Params,
    pair_to_index: Dict[Tuple[int, int], int],
    state_pairs: np.ndarray,
    max_size: int,
    clip_logabs: float = _LOG_FLUX_CLIP,
    closure_set: TwoStepClosureSet | None = None,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    closures = _resolve_closure_set(params, closure_set)
    i_flux: Dict[Tuple[int, int], float] = {}
    g_flux: Dict[Tuple[int, int], float] = {}
    k_flux: Dict[Tuple[int, int], float] = {}

    log_c1 = math.log(float(c1))
    w11 = float(w[pair_to_index[(1, 1)]])
    log_c_eq = log_c1 + w11 - w

    for idx, (i_raw, n_raw) in enumerate(state_pairs):
        i = int(i_raw)
        n = int(n_raw)

        next_i_val = f_flat[pair_to_index[(i + 1, n)]] if (i + 1) <= (max_size - 1) else 0.0
        i_sign, i_logabs = _signed_log_flux_term(
            math.log(f_attach(i, n, s_co, params, closure_set=closures)) + float(log_c_eq[idx]),
            float(f_flat[idx]),
            float(next_i_val),
        )
        i_flux[(i, n)] = _signed_log_to_float_clipped(i_sign, i_logabs, clip_logabs=clip_logabs)

        if n < i:
            next_g_val = f_flat[pair_to_index[(i, n + 1)]]
            g_sign, g_logabs = _signed_log_flux_term(
                math.log(g_attach(n, params, s_co=s_co, closure_set=closures)) + float(log_c_eq[idx]),
                float(f_flat[idx]),
                float(next_g_val),
            )
            g_flux[(i, n)] = _signed_log_to_float_clipped(g_sign, g_logabs, clip_logabs=clip_logabs)

        if n == i:
            next_k_val = f_flat[pair_to_index[(i + 1, i + 1)]] if (i + 1) <= (max_size - 1) else 0.0
            k_sign, k_logabs = _signed_log_flux_term(
                math.log(k_attach(i, s_co, params, closure_set=closures)) + float(log_c_eq[idx]),
                float(f_flat[idx]),
                float(next_k_val),
            )
            k_flux[(i, i)] = _signed_log_to_float_clipped(k_sign, k_logabs, clip_logabs=clip_logabs)

    return i_flux, g_flux, k_flux


def rate_dict_from_fluxes(
    s_co: float,
    params: Paper2022Params,
    i_flux: Dict[Tuple[int, int], float],
    g_flux: Dict[Tuple[int, int], float],
    k_flux: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    critical = critical_sizes(s_co, params)
    i_star = critical["i_star"]
    n_star = critical["n_star"]
    i_co_star = critical["i_co_star"]
    upper = params.max_size

    def I(i: int, n: int) -> float:
        return i_flux.get((i, n), 0.0)

    def G(i: int, n: int) -> float:
        return g_flux.get((i, n), 0.0)

    def K(i: int) -> float:
        return k_flux.get((i, i), 0.0)

    j_c = K(i_co_star) + sum(G(i, i - 1) - I(i, i) for i in range(i_co_star + 1, upper))
    j_c_d = sum(I(i, i) for i in range(n_star + 1, upper)) + sum(
        G(i, n_star) - G(i, i - 1) for i in range(n_star + 1, upper)
    )
    j_d = I(i_star, 1) - sum(G(i, 1) for i in range(i_star + 1, upper))

    j_d_plus_com = sum(I(i_star, n) for n in range(1, i_star + 1)) + sum(
        I(i, i) - G(i, i - 1) for i in range(i_star + 1, upper)
    )
    j_com = sum(I(i_star, n) for n in range(2, i_star + 1)) + sum(
        I(i, i) - G(i, i - 1) for i in range(i_star + 1, upper)
    )

    j_tot = K(i_co_star) + sum(I(i_co_star, n) for n in range(1, i_co_star + 1))
    j_comp = sum(I(i_co_star, n) for n in range(1, i_co_star + 1)) + sum(
        I(i, i) - G(i, i - 1) for i in range(i_co_star + 1, upper)
    )
    j_cr_met = sum(I(i, i) for i in range(n_star + 1, upper)) + sum(
        G(i, n_star) - G(i, i - 1) for i in range(n_star + 1, upper)
    )
    j_met = I(i_star, 1) - sum(G(i, 1) for i in range(i_star + 1, upper))

    return {
        "J_c": j_c,
        "J_c_d": j_c_d,
        "J_d": j_d,
        "J_d_plus_com": j_d_plus_com,
        "J_com": j_com,
        "J_tot": j_tot,
        "J_comp": j_comp,
        "J_cr_met": j_cr_met,
        "J_met": j_met,
    }


def rate_time_series_from_solution(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
) -> Dict[str, np.ndarray]:
    solution = solve_output["solution"]
    c_eq = solve_output["c_eq"]
    pair_to_index = solve_output["pair_to_index"]
    state_pairs = solve_output["state_pairs"]
    closures = _closure_set_from_solve_output(solve_output, None)

    names = ["J_c", "J_c_d", "J_d", "J_d_plus_com", "J_com", "J_tot", "J_comp", "J_cr_met", "J_met"]
    curves = {name: np.zeros(solution.y.shape[1], dtype=float) for name in names}

    for idx in range(solution.y.shape[1]):
        i_flux, g_flux, k_flux = compute_flux_maps(
            solution.y[:, idx],
            c_eq,
            s_co,
            params,
            pair_to_index,
            state_pairs,
            params.max_size,
            closure_set=closures,
        )
        rate_map = rate_dict_from_fluxes(s_co, params, i_flux, g_flux, k_flux)
        for name in names:
            curves[name][idx] = rate_map[name]

    return curves


def rate_time_series_from_solution_stable(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
) -> Dict[str, np.ndarray]:
    solution = solve_output["solution"]
    closures = _closure_set_from_solve_output(solve_output, None)
    names = ["J_c", "J_c_d", "J_d", "J_d_plus_com", "J_com", "J_tot", "J_comp", "J_cr_met", "J_met"]
    curves = {name: np.zeros(solution.y.shape[1], dtype=float) for name in names}

    for idx in range(solution.y.shape[1]):
        rate_map = rate_dict_from_state_stable(
            solution.y[:, idx],
            solve_output["w"],
            solve_output["c1"],
            s_co,
            params,
            solve_output["pair_to_index"],
            closure_set=closures,
        )
        for name in names:
            curves[name][idx] = rate_map[name]

    return curves


def rate_time_series_diagnostics_from_solution_stable(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
) -> Dict[str, Dict[str, np.ndarray]]:
    solution = solve_output["solution"]
    closures = _closure_set_from_solve_output(solve_output, None)
    names = ["J_c", "J_c_d", "J_d", "J_d_plus_com", "J_com", "J_tot", "J_comp", "J_cr_met", "J_met"]
    fields = [
        "value",
        "sign",
        "logabs",
        "abs_logsum",
        "positive_logsum",
        "negative_logsum",
        "cancellation_ratio",
        "term_count",
    ]
    curves = {
        name: {field: np.zeros(solution.y.shape[1], dtype=float) for field in fields}
        for name in names
    }

    for idx in range(solution.y.shape[1]):
        diagnostics = rate_dict_from_state_signed_stable(
            solution.y[:, idx],
            solve_output["w"],
            solve_output["c1"],
            s_co,
            params,
            solve_output["pair_to_index"],
            closure_set=closures,
        )
        for name in names:
            for field in fields:
                curves[name][field][idx] = float(diagnostics[name][field])

    return curves


def number_density_time_series(
    time_s: np.ndarray,
    rate_curves: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        name.replace("J_", "N_"): cumulative_trapezoid(values, time_s)
        for name, values in rate_curves.items()
    }


def flat_to_triangular_grid(
    flat_values: np.ndarray,
    state_pairs: np.ndarray,
    max_size: int,
    fill_value: float = np.nan,
) -> np.ndarray:
    grid = np.full((max_size, max_size), fill_value, dtype=float)
    for value, (i_raw, n_raw) in zip(flat_values, state_pairs):
        i = int(i_raw)
        n = int(n_raw)
        grid[n, i] = float(value)
    return grid


def free_energy_grid(
    s_co: float,
    params: Paper2022Params,
    max_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = max_size or params.max_size
    pair_to_index, state_pairs = triangular_index_maps(size)
    w, _, _ = make_w_and_c_arrays(s_co, params, state_pairs)
    grid = flat_to_triangular_grid(w, state_pairs, size)
    return np.arange(1, size), np.arange(1, size), grid


def extract_profiles(
    z_flat: np.ndarray,
    state_pairs: np.ndarray,
    max_size: int,
    i_star: int,
    n_star: int,
) -> Dict[str, np.ndarray]:
    grid = flat_to_triangular_grid(z_flat, state_pairs, max_size)
    i_axis = np.arange(1, max_size)
    n_axis = np.arange(1, max_size)

    diag = np.array([grid[i, i] for i in range(1, max_size)], dtype=float)
    droplets = np.array([grid[1, i] for i in range(1, max_size)], dtype=float)

    composite_n_star = np.array(
        [grid[n_star, i] if i > n_star else np.nan for i in range(1, max_size)],
        dtype=float,
    )
    crystal_i_star = np.array(
        [grid[n, i_star] if n <= i_star else np.nan for n in range(1, max_size)],
        dtype=float,
    )

    return {
        "i_axis": i_axis,
        "n_axis": n_axis,
        "Z_i_i": diag,
        "Z_i_1": droplets,
        "Z_i_n_star": composite_n_star,
        "Z_i_star_n": crystal_i_star,
    }


def stationary_reference_rates(s_co: float, params: Paper2022Params) -> Dict[str, float]:
    pair_to_index, state_pairs = triangular_index_maps(params.max_size)
    w, _, c1 = make_w_and_c_arrays(s_co, params, state_pairs)
    critical = critical_sizes(s_co, params)
    i_star = critical["i_star"]
    n_star = critical["n_star"]
    i_co_star = critical["i_co_star"]
    w11 = w_total(1, 1, s_co, params)

    def w_pair(i: int, n: int) -> float:
        return w[pair_to_index[(i, n)]]

    j_c_1s = c1 * params.f0 / sum(
        np.exp(w_pair(i, i) - w11) / (1.0 * i ** (2.0 / 3.0) * np.exp(s_co))
        for i in range(1, params.max_size)
    )
    j_d_1s = c1 * params.f0 / sum(
        np.exp(w_pair(i, 1) - w11) / (i ** (2.0 / 3.0) * np.exp(s_co))
        for i in range(1, params.max_size)
    )

    theta_c_1s = (8.0 * params.gamma_co / (3.0 * params.f0 * s_co**2)) * np.exp(-s_co)
    theta_d_1s = (
        8.0 * params.gamma_mo / (3.0 * params.f0 * (s_co - params.s_cm) ** 2)
    ) * np.exp(-s_co)
    theta_c_d_inf = (8.0 * params.gamma_cm / (3.0 * params.g0 * params.s_cm**2)) * np.exp(-params.s_cm)

    return {
        "J_c_1S": j_c_1s,
        "J_d_1S": j_d_1s,
        "theta_c_1S_us": theta_c_1s * 1.0e6,
        "theta_d_1S_us": theta_d_1s * 1.0e6,
        "theta_c_d_inf_us": theta_c_d_inf * 1.0e6,
        "i_star": float(i_star),
        "n_star": float(n_star),
        "i_co_star": float(i_co_star),
    }


def stationary_bulk_crystal_rate(params: Paper2022Params) -> float:
    n = np.arange(1, params.max_size, dtype=float)
    w_cm = -params.s_cm * n + params.gamma_cm * n ** (2.0 / 3.0)
    g_n = params.g0 * np.exp(params.s_cm) * n ** (2.0 / 3.0)
    c0 = params.c1_base
    return c0 / np.sum(np.exp(w_cm) / g_n)


def crystal_rate_components(
    s_co: float,
    params: Paper2022Params,
    i_flux: Dict[Tuple[int, int], float],
    g_flux: Dict[Tuple[int, int], float],
    k_flux: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    critical = critical_sizes(s_co, params)
    i_co_star = critical["i_co_star"]

    k_component = k_flux.get((i_co_star, i_co_star), 0.0)
    g_component = sum(g_flux.get((i, i - 1), 0.0) for i in range(i_co_star + 1, params.max_size))
    i_component = sum(i_flux.get((i, i), 0.0) for i in range(i_co_star + 1, params.max_size))

    return {
        "K": k_component,
        "G": g_component,
        "I": i_component,
        "J_c_from_components": k_component + g_component - i_component,
    }


def crystal_rate_components_from_solve_output_stable(
    solve_output: Dict[str, object],
    s_co: float,
    params: Paper2022Params,
) -> Dict[str, object]:
    f_flat = solve_output["solution"].y[:, -1]
    closures = _closure_set_from_solve_output(solve_output, None)
    return decompose_j_c_terms_stable(
        f_flat,
        solve_output["w"],
        solve_output["c1"],
        s_co,
        params,
        solve_output["pair_to_index"],
        closure_set=closures,
    )


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(y_arr, dtype=float)
    increments = 0.5 * (y_arr[1:] + y_arr[:-1]) * np.diff(x_arr)
    total = 0.0
    compensation = 0.0
    for idx, increment in enumerate(increments, start=1):
        corrected = float(increment) - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
        out[idx] = total
    return out


def delay_time_from_rate_and_density(
    time_s: np.ndarray,
    rate: np.ndarray,
    density: np.ndarray,
    tail_fraction: float = 0.25,
    min_tail_points: int = 5,
) -> float:
    time_arr = np.asarray(time_s, dtype=float)
    rate_arr = np.asarray(rate, dtype=float)
    density_arr = np.asarray(density, dtype=float)

    finite = np.isfinite(time_arr) & np.isfinite(rate_arr) & np.isfinite(density_arr)
    if int(np.count_nonzero(finite)) < 2:
        return float("nan")

    time_valid = time_arr[finite]
    rate_valid = rate_arr[finite]
    density_valid = density_arr[finite]
    order = np.argsort(time_valid)
    time_valid = time_valid[order]
    rate_valid = rate_valid[order]
    density_valid = density_valid[order]

    tail_count = max(int(min_tail_points), int(math.ceil(len(time_valid) * float(tail_fraction))))
    tail_count = min(tail_count, len(time_valid))
    tail_time = time_valid[-tail_count:]
    tail_density = density_valid[-tail_count:]

    if len(tail_time) >= 2 and np.ptp(tail_time) > 0.0:
        centered_time = tail_time - float(np.mean(tail_time))
        design = np.column_stack([centered_time, np.ones_like(centered_time)])
        slope, intercept_centered = np.linalg.lstsq(design, tail_density, rcond=None)[0]
        if np.isfinite(slope) and slope != 0.0:
            intercept = float(intercept_centered) - float(slope) * float(np.mean(tail_time))
            return float(-intercept / slope)

    stationary_rate = float(rate_valid[-1])
    if stationary_rate == 0.0 or not math.isfinite(stationary_rate):
        return float("nan")
    return float(time_valid[-1] - density_valid[-1] / stationary_rate)
