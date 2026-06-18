#!/usr/bin/env python3
# surface_mismatch.py

"""
===============================================================================
Surface mismatch calculator
===============================================================================

Purpose
-------
두 결정 구조 사이에서 가능한 low-index surface pair를 탐색하고,
각 surface pair의 2D in-plane lattice mismatch를 계산한다.

즉,

    structure 1의 (hkl)_1  ||  structure 2의 (hkl)_2

관계에서 두 surface lattice가 얼마나 잘 맞는지 screening한다.

이 코드는 interface energy, termination, chemistry, defect, strain relaxation을
계산하지 않는다. 따라서 결과는 다음 의미로 해석해야 한다.

    "두 phase 사이에 crystallographically plausible한
     low-mismatch orientation relationship 후보가 있는가?"

Run
---
코드 상단의 USER CONFIG만 수정한 뒤 실행한다.

    python surface_mismatch.py

Input modes
-----------
각 구조는 아래 두 방식 중 하나로 입력할 수 있다.

1. local mode
   - DFT relaxation 결과인 CONTCAR 또는 POSCAR 파일을 사용한다.
   - 파일 경로 또는 계산 폴더 경로를 넣을 수 있다.
   - 폴더 경로를 넣으면 내부에서 CONTCAR, POSCAR를 자동 검색한다.

   Example:
       STRUCTURE_1_SOURCE = "local"
       STRUCTURE_1_LOCAL_PATH = "/home/user/wurtzite/CONTCAR"

2. mp mode
   - Materials Project mp-id를 사용한다.
   - MP_API_KEY가 필요하다.

   Example:
       STRUCTURE_1_SOURCE = "mp"
       STRUCTURE_1_MP_ID = "mp-149"
       MP_API_KEY = "your_api_key_here"

Recommended use cases
---------------------
1. DFT-relaxed polymorph mismatch

       STRUCTURE_1_SOURCE = "local"
       STRUCTURE_2_SOURCE = "local"

       STRUCTURE_1_LOCAL_PATH = "/path/to/phase_A/CONTCAR"
       STRUCTURE_2_LOCAL_PATH = "/path/to/phase_B/CONTCAR"

       STANDARDIZE_STRUCTURE = "none"

2. Materials Project structure mismatch

       STRUCTURE_1_SOURCE = "mp"
       STRUCTURE_2_SOURCE = "mp"

       STRUCTURE_1_MP_ID = "mp-XXXX"
       STRUCTURE_2_MP_ID = "mp-YYYY"

       MP_API_KEY = "your_api_key_here"
       STANDARDIZE_STRUCTURE = "conventional"

3. Mixed mode

       STRUCTURE_1_SOURCE = "local"
       STRUCTURE_2_SOURCE = "mp"

Important parameters
--------------------
MAX_INDEX
    조사할 최대 Miller index.
    MAX_INDEX = 2이면 (100), (110), (111), (210), (211), ... 등을 조사한다.

MAX_DET
    허용할 2D surface supercell의 최대 determinant.
    값이 클수록 domain matching을 더 많이 찾지만 계산 시간이 증가한다.
    작은 coherent-like matching만 보고 싶으면 4~8 정도 권장.
    큰 domain matching까지 보고 싶으면 12~20 정도까지 증가.

COEFF_RANGE
    2D supercell matrix의 정수 계수 범위.
    보통 MAX_DET = 8이면 COEFF_RANGE = 4~5 정도면 충분하다.

HKL_LIST_1, HKL_LIST_2
    None이면 자동으로 low-index hkl을 생성한다.
    특정 면만 보고 싶으면 아래처럼 직접 지정한다.

        HKL_LIST_1 = [(0, 0, 1), (1, 0, 0)]
        HKL_LIST_2 = [(1, 1, 1), (1, 1, 0)]

Output
------
OUTPUT_CSV 파일에 결과가 저장된다.

중요한 column:
    hkl_1, hkl_2
        비교한 surface pair

    score_percent
        전체 mismatch score. 낮을수록 좋다.

    length1_mismatch_percent, length2_mismatch_percent
        두 in-plane lattice vector 길이 mismatch

    area_mismatch_percent
        2D surface cell area mismatch

    angle_mismatch_deg
        두 2D surface cell의 angle mismatch

    det_1, det_2
        각 구조에서 사용한 surface supercell determinant.
        det가 작을수록 더 단순한 matching이다.

Interpretation guide
--------------------
대략적 기준:

    score < 2%:
        매우 좋은 geometric matching 후보

    2% < score < 5%:
        plausible matching 후보

    5% < score < 10%:
        strain/domain matching이 필요할 수 있음

    score > 10%:
        coherent transition/interface로는 부담이 큼

주의:
    낮은 mismatch가 큰 det에서만 나오면 coherent interface라기보다
    domain-matched interface로 해석하는 것이 안전하다.

Dependencies
------------
    pip install numpy pandas pymatgen mp-api

===============================================================================
"""

import os
from pathlib import Path
from itertools import product
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# =============================================================================
# USER CONFIG
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Input source
# -----------------------------------------------------------------------------
# Options:
#   "local" : read CONTCAR/POSCAR from local path
#   "mp"    : read structure from Materials Project
# -----------------------------------------------------------------------------

STRUCTURE_1_SOURCE = "local"
STRUCTURE_2_SOURCE = "local"


# -----------------------------------------------------------------------------
# 2. Local structure paths
# -----------------------------------------------------------------------------
# If source is "local", set path here.
# You can use either:
#   - direct file path: /path/to/CONTCAR
#   - directory path:  /path/to/relax_folder/
#
# If a directory is given, the code searches:
#   CONTCAR -> contcar -> POSCAR -> poscar
# -----------------------------------------------------------------------------

STRUCTURE_1_LOCAL_PATH = "/home/user/phase_1/CONTCAR"
STRUCTURE_2_LOCAL_PATH = "/home/user/phase_2/CONTCAR"


# -----------------------------------------------------------------------------
# 3. Materials Project settings
# -----------------------------------------------------------------------------
# If source is "mp", set mp-id and API key here.
#
# Important:
#   Do not upload or commit your real API key to public repositories.
# -----------------------------------------------------------------------------

MP_API_KEY = "PASTE_YOUR_MATERIALS_PROJECT_API_KEY_HERE"

STRUCTURE_1_MP_ID = "mp-XXXX"
STRUCTURE_2_MP_ID = "mp-YYYY"


# -----------------------------------------------------------------------------
# 4. Structure standardization
# -----------------------------------------------------------------------------
# For DFT-relaxed CONTCAR:
#   "none" is recommended.
#
# For Materials Project structures:
#   "conventional" is often convenient.
#
# Options:
#   "none"
#   "primitive"
#   "conventional"
# -----------------------------------------------------------------------------

STANDARDIZE_STRUCTURE = "none"
SYMPREC = 0.1


# -----------------------------------------------------------------------------
# 5. Surface search settings
# -----------------------------------------------------------------------------

MAX_INDEX = 2
MAX_DET = 8
COEFF_RANGE = 4
PLANE_SEARCH_RANGE = 6

# Optional maximum 2D surface supercell area in Å^2.
# Use None for no cutoff.
MAX_AREA = None

# If None, hkl list is automatically generated using MAX_INDEX.
# If you want to test only selected planes, use list of tuples.
#
# Example:
#   HKL_LIST_1 = [(0, 0, 1), (1, 0, 0)]
#   HKL_LIST_2 = [(1, 1, 1), (1, 1, 0)]
HKL_LIST_1 = None
HKL_LIST_2 = None


# -----------------------------------------------------------------------------
# 6. Output settings
# -----------------------------------------------------------------------------

TOP_N = 100
OUTPUT_CSV = "surface_mismatch_results.csv"

# Print all rows if False; print only TOP_N rows if True.
SAVE_ONLY_TOP_N = True


# =============================================================================
# Structure loading
# =============================================================================

def clean_api_key(api_key):
    if api_key is None:
        return None
    api_key = str(api_key).strip()
    if api_key == "":
        return None
    if "PASTE_YOUR" in api_key or "YOUR_MATERIALS_PROJECT_API_KEY" in api_key:
        return None
    return api_key


def resolve_structure_file(path_str):
    p = Path(path_str).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        for name in ["CONTCAR", "contcar", "POSCAR", "poscar"]:
            candidate = p / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Cannot find CONTCAR/POSCAR from: {path_str}")


def load_local_structure(path_str):
    path = resolve_structure_file(path_str)
    structure = Structure.from_file(str(path))
    label = str(path)
    return structure, label


def load_mp_structure(mp_id, api_key):
    api_key = clean_api_key(api_key) or clean_api_key(os.environ.get("MP_API_KEY"))

    if api_key is None:
        raise ValueError(
            "Materials Project API key is missing. "
            "Set MP_API_KEY in USER CONFIG or export MP_API_KEY."
        )

    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        structure = mpr.get_structure_by_material_id(mp_id)

    label = mp_id
    return structure, label


def standardize_structure(structure, mode="none", symprec=0.1):
    if mode == "none":
        return structure

    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)

        if mode == "primitive":
            return sga.get_primitive_standard_structure()

        if mode == "conventional":
            return sga.get_conventional_standard_structure()

    except Exception as exc:
        print(f"[WARNING] Structure standardization failed: {exc}")
        print("[WARNING] Using original structure.")

    return structure


def load_structure(source, local_path, mp_id, api_key):
    source = source.lower().strip()

    if source == "local":
        structure, label = load_local_structure(local_path)

    elif source == "mp":
        structure, label = load_mp_structure(mp_id, api_key)

    else:
        raise ValueError(f"Unknown source type: {source}. Use 'local' or 'mp'.")

    structure = standardize_structure(
        structure,
        mode=STANDARDIZE_STRUCTURE,
        symprec=SYMPREC,
    )

    return structure, label


# =============================================================================
# Miller index generation
# =============================================================================

def normalize_hkl(hkl):
    h, k, l = hkl
    g = reduce(gcd, [abs(h), abs(k), abs(l)])

    if g == 0:
        return None

    h, k, l = h // g, k // g, l // g

    # Canonical sign: first nonzero index should be positive.
    for x in (h, k, l):
        if x != 0:
            if x < 0:
                h, k, l = -h, -k, -l
            break

    return (h, k, l)


def generate_hkls(max_index):
    hkls = set()

    for h, k, l in product(
        range(-max_index, max_index + 1),
        range(-max_index, max_index + 1),
        range(-max_index, max_index + 1),
    ):
        if (h, k, l) == (0, 0, 0):
            continue

        nhkl = normalize_hkl((h, k, l))
        if nhkl is not None:
            hkls.add(nhkl)

    return sorted(
        hkls,
        key=lambda x: (max(abs(i) for i in x), sum(abs(i) for i in x), x),
    )


def get_hkl_list(user_hkl_list):
    if user_hkl_list is None:
        return generate_hkls(MAX_INDEX)

    cleaned = []
    for hkl in user_hkl_list:
        nhkl = normalize_hkl(tuple(hkl))
        if nhkl is not None and nhkl not in cleaned:
            cleaned.append(nhkl)

    return cleaned


# =============================================================================
# Surface 2D lattice construction
# =============================================================================

def vector_from_integer_coeffs(lattice_matrix, coeffs):
    """
    pymatgen lattice.matrix uses row-vector convention:
        lattice_matrix[0] = a vector
        lattice_matrix[1] = b vector
        lattice_matrix[2] = c vector
    """
    coeffs = np.array(coeffs, dtype=float)
    return coeffs @ lattice_matrix


def surface_basis_from_hkl(lattice_matrix, hkl, search_range=6, tol=1e-8):
    """
    Find two short non-collinear lattice vectors lying in the (hkl) plane.

    Direct lattice vector:
        T = u*a + v*b + w*c

    Plane condition:
        h*u + k*v + l*w = 0
    """
    h, k, l = hkl
    candidates = []

    for u, v, w in product(
        range(-search_range, search_range + 1),
        range(-search_range, search_range + 1),
        range(-search_range, search_range + 1),
    ):
        if (u, v, w) == (0, 0, 0):
            continue

        if h * u + k * v + l * w == 0:
            coeff = np.array([u, v, w], dtype=int)
            vec = vector_from_integer_coeffs(lattice_matrix, coeff)
            length = np.linalg.norm(vec)
            candidates.append((length, coeff, vec))

    if len(candidates) < 2:
        raise RuntimeError(f"Not enough in-plane vectors found for hkl={hkl}")

    candidates.sort(key=lambda x: x[0])

    best_pair = None
    best_area = np.inf

    for _, c1, v1 in candidates:
        for _, c2, v2 in candidates:
            area = np.linalg.norm(np.cross(v1, v2))

            if area < tol:
                continue

            if area < best_area:
                best_area = area
                best_pair = (c1, c2, v1, v2)

    if best_pair is None:
        raise RuntimeError(f"No non-collinear basis found for hkl={hkl}")

    c1, c2, v1, v2 = best_pair

    return {
        "basis_frac": np.array([c1, c2], dtype=int),
        "basis_cart": np.array([v1, v2], dtype=float),
        "area": best_area,
    }


# =============================================================================
# 2D lattice descriptor
# =============================================================================

def gauss_reduce_2d(basis, max_iter=100):
    v1 = basis[0].astype(float).copy()
    v2 = basis[1].astype(float).copy()

    for _ in range(max_iter):
        changed = False

        if np.dot(v2, v2) < np.dot(v1, v1):
            v1, v2 = v2, v1
            changed = True

        mu = np.round(np.dot(v1, v2) / np.dot(v1, v1))
        if abs(mu) > 0:
            v2 = v2 - mu * v1
            changed = True

        if not changed:
            break

    return np.array([v1, v2], dtype=float)


def descriptor_2d(basis):
    """
    Returns:
        l1, l2, acute_angle_deg, area
    """
    red = gauss_reduce_2d(basis)
    v1, v2 = red

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)

    cosang = np.dot(v1, v2) / (l1 * l2)
    cosang = np.clip(cosang, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosang))

    if angle > 90:
        angle = 180 - angle

    area = np.linalg.norm(np.cross(v1, v2))

    if l1 > l2:
        l1, l2 = l2, l1

    return np.array([l1, l2, angle, area], dtype=float)


# =============================================================================
# 2D supercell enumeration
# =============================================================================

def enumerate_2d_supercells(
    basis_cart,
    max_det=8,
    coeff_range=4,
    max_area=None,
):
    """
    Generate 2D supercells:
        B_super = M @ B_surface

    M is a 2x2 integer matrix.
    |det(M)| is the surface area multiplier.
    """
    cells = []

    for a, b, c, d in product(
        range(-coeff_range, coeff_range + 1),
        range(-coeff_range, coeff_range + 1),
        range(-coeff_range, coeff_range + 1),
        range(-coeff_range, coeff_range + 1),
    ):
        M = np.array([[a, b], [c, d]], dtype=int)
        det = int(round(np.linalg.det(M)))
        det_abs = abs(det)

        if det_abs == 0 or det_abs > max_det:
            continue

        sc_basis = M @ basis_cart
        area = np.linalg.norm(np.cross(sc_basis[0], sc_basis[1]))

        if area < 1e-8:
            continue

        if max_area is not None and area > max_area:
            continue

        desc = descriptor_2d(sc_basis)

        cells.append({
            "M": M,
            "det": det_abs,
            "basis": sc_basis,
            "descriptor": desc,
        })

    # Remove near-duplicate descriptors.
    unique = {}
    for cell in cells:
        key = tuple(np.round(cell["descriptor"], 5))
        if key not in unique:
            unique[key] = cell

    return list(unique.values())


# =============================================================================
# Mismatch metric
# =============================================================================

def compare_descriptors(desc1, desc2):
    l1a, l2a, anga, areaa = desc1
    l1b, l2b, angb, areab = desc2

    f_l1 = abs(l1a - l1b) / ((l1a + l1b) / 2)
    f_l2 = abs(l2a - l2b) / ((l2a + l2b) / 2)
    f_area = abs(areaa - areab) / ((areaa + areab) / 2)
    f_angle_deg = abs(anga - angb)

    rms_length_mismatch = np.sqrt((f_l1**2 + f_l2**2) / 2)

    # Conservative score.
    # 1 degree angle mismatch is treated roughly as 1% mismatch.
    score = max(f_l1, f_l2, f_area, f_angle_deg / 100.0)

    return score, {
        "length1_mismatch": f_l1,
        "length2_mismatch": f_l2,
        "rms_length_mismatch": rms_length_mismatch,
        "area_mismatch": f_area,
        "angle_mismatch_deg": f_angle_deg,
    }


# =============================================================================
# Surface cell precomputation
# =============================================================================

def precompute_surface_cells(
    structure,
    hkls,
    plane_search_range,
    max_det,
    coeff_range,
    max_area,
):
    lattice_matrix = structure.lattice.matrix
    data = {}

    for hkl in hkls:
        try:
            surf = surface_basis_from_hkl(
                lattice_matrix,
                hkl,
                search_range=plane_search_range,
            )

            cells = enumerate_2d_supercells(
                surf["basis_cart"],
                max_det=max_det,
                coeff_range=coeff_range,
                max_area=max_area,
            )

            if len(cells) > 0:
                data[hkl] = {
                    "surface_basis_frac": surf["basis_frac"],
                    "surface_basis_cart": surf["basis_cart"],
                    "surface_area": surf["area"],
                    "cells": cells,
                }

        except Exception as exc:
            print(f"[WARNING] Failed hkl={hkl}: {exc}")

    return data


# =============================================================================
# Main mismatch scan
# =============================================================================

def scan_mismatch(struct1, struct2, label1, label2):
    hkls1 = get_hkl_list(HKL_LIST_1)
    hkls2 = get_hkl_list(HKL_LIST_2)

    print(f"[INFO] Number of hkl candidates for structure 1: {len(hkls1)}")
    print(f"[INFO] Number of hkl candidates for structure 2: {len(hkls2)}")

    print("[INFO] Precomputing surface cells for structure 1...")
    surf1 = precompute_surface_cells(
        struct1,
        hkls1,
        plane_search_range=PLANE_SEARCH_RANGE,
        max_det=MAX_DET,
        coeff_range=COEFF_RANGE,
        max_area=MAX_AREA,
    )

    print("[INFO] Precomputing surface cells for structure 2...")
    surf2 = precompute_surface_cells(
        struct2,
        hkls2,
        plane_search_range=PLANE_SEARCH_RANGE,
        max_det=MAX_DET,
        coeff_range=COEFF_RANGE,
        max_area=MAX_AREA,
    )

    rows = []

    for hkl1, data1 in surf1.items():
        for hkl2, data2 in surf2.items():
            best_score = np.inf
            best_detail = None
            best_pair = None

            for c1 in data1["cells"]:
                for c2 in data2["cells"]:
                    score, detail = compare_descriptors(
                        c1["descriptor"],
                        c2["descriptor"],
                    )

                    if score < best_score:
                        best_score = score
                        best_detail = detail
                        best_pair = (c1, c2)

            if best_pair is None:
                continue

            c1, c2 = best_pair

            rows.append({
                "structure_1": label1,
                "structure_2": label2,
                "formula_1": struct1.composition.reduced_formula,
                "formula_2": struct2.composition.reduced_formula,

                "hkl_1": str(hkl1),
                "hkl_2": str(hkl2),

                "score_percent": 100 * best_score,
                "length1_mismatch_percent": 100 * best_detail["length1_mismatch"],
                "length2_mismatch_percent": 100 * best_detail["length2_mismatch"],
                "rms_length_mismatch_percent": 100 * best_detail["rms_length_mismatch"],
                "area_mismatch_percent": 100 * best_detail["area_mismatch"],
                "angle_mismatch_deg": best_detail["angle_mismatch_deg"],

                "det_1": c1["det"],
                "det_2": c2["det"],

                "M_1": c1["M"].tolist(),
                "M_2": c2["M"].tolist(),

                "cell_1_l1_A": c1["descriptor"][0],
                "cell_1_l2_A": c1["descriptor"][1],
                "cell_1_angle_deg": c1["descriptor"][2],
                "cell_1_area_A2": c1["descriptor"][3],

                "cell_2_l1_A": c2["descriptor"][0],
                "cell_2_l2_A": c2["descriptor"][1],
                "cell_2_angle_deg": c2["descriptor"][2],
                "cell_2_area_A2": c2["descriptor"][3],

                "surface_basis_frac_1": data1["surface_basis_frac"].tolist(),
                "surface_basis_frac_2": data2["surface_basis_frac"].tolist(),
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values("score_percent").reset_index(drop=True)

    if SAVE_ONLY_TOP_N:
        df = df.head(TOP_N)

    return df


# =============================================================================
# Execution
# =============================================================================

def print_header():
    print("=" * 80)
    print("Surface mismatch calculation")
    print("=" * 80)


def print_structure_info(index, source, label, structure):
    print(f"\n[STRUCTURE {index}]")
    print("source  :", source)
    print("label   :", label)
    print("formula :", structure.composition.reduced_formula)
    print("sites   :", len(structure))
    print("lattice :")
    print(structure.lattice)


def print_settings():
    print("\n[SEARCH SETTINGS]")
    print("STANDARDIZE_STRUCTURE :", STANDARDIZE_STRUCTURE)
    print("SYMPREC               :", SYMPREC)
    print("MAX_INDEX             :", MAX_INDEX)
    print("MAX_DET               :", MAX_DET)
    print("COEFF_RANGE           :", COEFF_RANGE)
    print("PLANE_SEARCH_RANGE    :", PLANE_SEARCH_RANGE)
    print("MAX_AREA              :", MAX_AREA)
    print("HKL_LIST_1            :", HKL_LIST_1)
    print("HKL_LIST_2            :", HKL_LIST_2)
    print("TOP_N                 :", TOP_N)
    print("SAVE_ONLY_TOP_N       :", SAVE_ONLY_TOP_N)
    print("OUTPUT_CSV            :", OUTPUT_CSV)


def main():
    print_header()

    print("[INFO] Loading structure 1...")
    struct1, label1 = load_structure(
        STRUCTURE_1_SOURCE,
        STRUCTURE_1_LOCAL_PATH,
        STRUCTURE_1_MP_ID,
        MP_API_KEY,
    )

    print("[INFO] Loading structure 2...")
    struct2, label2 = load_structure(
        STRUCTURE_2_SOURCE,
        STRUCTURE_2_LOCAL_PATH,
        STRUCTURE_2_MP_ID,
        MP_API_KEY,
    )

    print_structure_info(1, STRUCTURE_1_SOURCE, label1, struct1)
    print_structure_info(2, STRUCTURE_2_SOURCE, label2, struct2)
    print_settings()

    df = scan_mismatch(struct1, struct2, label1, label2)

    if len(df) == 0:
        print("\n[RESULT] No matches found.")
        return

    df.to_csv(OUTPUT_CSV, index=False)

    show_cols = [
        "hkl_1",
        "hkl_2",
        "score_percent",
        "length1_mismatch_percent",
        "length2_mismatch_percent",
        "rms_length_mismatch_percent",
        "area_mismatch_percent",
        "angle_mismatch_deg",
        "det_1",
        "det_2",
        "M_1",
        "M_2",
    ]

    print("\n[TOP MATCHES]")
    print(df[show_cols].head(TOP_N).to_string(index=False))

    print(f"\n[INFO] Saved CSV to: {OUTPUT_CSV}")

    print("\n[INTERPRETATION]")
    print("Lower score_percent indicates better 2D geometric matching.")
    print("Small det_1 and det_2 indicate simpler, more coherent-like matching.")
    print("Low mismatch only at large determinants suggests domain matching rather than a simple coherent interface.")


if __name__ == "__main__":
    main()