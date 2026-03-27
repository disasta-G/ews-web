#!/usr/bin/env python3
"""
Validate g-function calculations against reference values from Ausgabe.ews.

Tests:
1. FLS image g-function basic properties
2. Single-borehole g-function
3. Borehole field g-function (4 project + 2 neighbor probes)
4. Comparison with stored reference g-values from the .ews file
5. Interpolation correctness
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from engine.ews_parser import get_simulation_params
from engine.g_functions import (
    t_s, g_eskilson_single, g_carslaw_jaeger, g_fls_image,
    g_field, compute_g_values, interpolate_g
)

EWS_FILE = os.path.join(os.path.dirname(__file__), 'tests', 'fixtures', 'Ausgabe.ews')


def main():
    # --- Load parameters ---
    params = get_simulation_params(EWS_FILE)

    H = params['H']
    r1 = params['r1']
    lambda_earth = params['lambda_earth']
    rho_earth = params['rho_earth']
    cp_earth = params['cp_earth']
    a = lambda_earth / (rho_earth * cp_earth)  # thermal diffusivity
    positions = params['positions']
    neighbor_positions = params['neighbor_positions']
    g_ref = params['g_values']

    print("=" * 72)
    print("G-FUNCTION VALIDATION against Ausgabe.ews")
    print("=" * 72)
    print()
    print("Borehole length H      = {} m".format(H))
    print("Borehole radius r1     = {} m".format(r1))
    print("lambda_earth           = {} W/mK".format(lambda_earth))
    print("Thermal diffusivity a  = {:.6e} m^2/s".format(a))
    print("Project probes         = {} at {}".format(len(positions), positions))
    print("Neighbor probes        = {} at {}".format(len(neighbor_positions), neighbor_positions))

    ts_val = t_s(H, a)
    print("t_s (Eskilson)         = {:.2f} s = {:.2f} yr".format(
        ts_val, ts_val / (3600 * 24 * 365.25)))
    print()

    # Reference g-values
    print("Reference g-values from Ausgabe.ews:")
    for ln_val in sorted(g_ref.keys()):
        print("  ln(t/ts) = {:+d}:  g = {:.13f}".format(ln_val, g_ref[ln_val]))
    print()

    ln_ts_values = [-4, -2, 0, 2, 3]
    all_tests_pass = True

    # =========================================================================
    # Test 1: FLS image basic properties
    # =========================================================================
    print("-" * 72)
    print("TEST 1: g_fls_image basic properties")
    print("-" * 72)

    # g(t=0) = 0
    assert g_fls_image(r1, H, 0, a) == 0.0, "g(t=0) should be 0"
    print("  g(t=0) = 0: PASS")

    # g monotonically increasing with time
    g_vals = [g_fls_image(r1, H, ts_val * np.exp(ln), a) for ln in ln_ts_values]
    for i in range(len(g_vals) - 1):
        assert g_vals[i] < g_vals[i+1], "g should increase with time"
    print("  g monotonically increasing: PASS")

    # g decreasing with distance
    t = ts_val
    g_r1 = g_fls_image(r1, H, t, a)
    g_5m = g_fls_image(5.0, H, t, a)
    g_10m = g_fls_image(10.0, H, t, a)
    assert g_r1 > g_5m > g_10m, "g should decrease with distance"
    print("  g decreasing with distance: PASS")

    # g converges at large times
    g_large1 = g_fls_image(r1, H, ts_val * np.exp(3), a)
    g_large2 = g_fls_image(r1, H, ts_val * np.exp(5), a)
    assert abs(g_large2 - g_large1) < 0.1, "g should converge"
    print("  g converges at large times: PASS")
    print()

    # =========================================================================
    # Test 2: Single borehole comparison
    # =========================================================================
    print("-" * 72)
    print("TEST 2: Single borehole g-function (FLS-image vs Eskilson)")
    print("-" * 72)

    print("  {:>10s}  {:>12s}  {:>14s}  {:>14s}".format(
        "ln(t/ts)", "t [yr]", "g_fls_image", "g_eskilson"))
    for ln_val in ln_ts_values:
        t = ts_val * np.exp(ln_val)
        t_yr = t / (3600 * 24 * 365.25)
        g_fls = g_fls_image(r1, H, t, a)
        g_esk = g_eskilson_single(r1, H, t, a)
        print("  {:+10d}  {:12.4f}  {:14.10f}  {:14.10f}".format(
            ln_val, t_yr, g_fls, g_esk))
    print()

    # =========================================================================
    # Test 3: Field g-function (4 project + 2 neighbor probes)
    # =========================================================================
    print("-" * 72)
    print("TEST 3: Field g-function (4 project + 2 neighbor probes)")
    print("-" * 72)

    g_field_values = compute_g_values(positions, r1, H, a, neighbor_positions)

    print("  {:>10s}  {:>12s}  {:>14s}  {:>14s}  {:>10s}  {:>8s}  {:>6s}".format(
        "ln(t/ts)", "t [yr]", "g_calc", "g_ref", "diff", "rel%", ""))
    for ln_val in ln_ts_values:
        g_calc = g_field_values[ln_val]
        g_reference = g_ref.get(ln_val, None)
        t = ts_val * np.exp(ln_val)
        t_yr = t / (3600 * 24 * 365.25)
        if g_reference is not None:
            diff = g_calc - g_reference
            rel = abs(diff) / g_reference * 100
            ok = rel < 2.0
            if not ok:
                all_tests_pass = False
            print("  {:+10d}  {:12.4f}  {:14.10f}  {:14.10f}  {:+10.6f}  {:8.4f}  {}".format(
                ln_val, t_yr, g_calc, g_reference, diff, rel,
                "PASS" if ok else "FAIL"))
        else:
            print("  {:+10d}  {:12.4f}  {:14.10f}  {:>14s}".format(
                ln_val, t_yr, g_calc, "N/A"))
    print()

    # =========================================================================
    # Test 4: Interpolation
    # =========================================================================
    print("-" * 72)
    print("TEST 4: g-value interpolation")
    print("-" * 72)

    # At table points
    for ln_val in sorted(g_ref.keys()):
        t = ts_val * np.exp(ln_val)
        g_interp = interpolate_g(t, H, a, g_ref, r1=r1)
        ref = g_ref[ln_val]
        err = abs(g_interp - ref)
        ok = err < 1e-6
        print("  ln={:+d}: interp={:.6f}  ref={:.6f}  diff={:.2e}  {}".format(
            ln_val, g_interp, ref, err, "PASS" if ok else "FAIL"))

    # Between table points
    t_mid = ts_val * np.exp(-1)
    g_mid = interpolate_g(t_mid, H, a, g_ref, r1=r1)
    expected = g_ref[-2] + (g_ref[0] - g_ref[-2]) * 0.5
    print("  ln=-1 (midpoint): interp={:.4f}  expected_linear={:.4f}".format(
        g_mid, expected))
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    max_err = 0
    mean_err = 0
    count = 0
    for ln_val in ln_ts_values:
        if ln_val in g_ref:
            err = abs(g_field_values[ln_val] - g_ref[ln_val])
            rel = err / g_ref[ln_val] * 100
            max_err = max(max_err, rel)
            mean_err += rel
            count += 1
    mean_err /= count if count else 1

    print("  Field g-function vs reference:")
    print("    Mean relative error: {:.4f}%".format(mean_err))
    print("    Max  relative error: {:.4f}%".format(max_err))

    if max_err < 2.0:
        print("    RESULT: PASS (all values within 2%)")
    else:
        print("    RESULT: FAIL (max error >= 2%)")
        all_tests_pass = False

    print()

    if all_tests_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
