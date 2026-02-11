#!/usr/bin/env python3
"""
Export PolyMixer coefficients to C++ header format.

Loads the sklearn pipeline from models/poly_model.pkl, extracts the
polynomial powers and linear regression coefficients, and writes them
as C arrays suitable for inclusion in a header-only library.

Also verifies the raw math reproduces sklearn's predictions.
"""

import pickle
import numpy as np
from pathlib import Path


def load_model(path="models/poly_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_features_manual(x, powers):
    """Reproduce sklearn PolynomialFeatures using the powers matrix."""
    features = np.ones(len(powers))
    for i, p in enumerate(powers):
        val = 1.0
        for j, exp in enumerate(p):
            if exp != 0:
                val *= x[j] ** exp
        features[i] = val
    return features


def verify(model, powers, coef, intercept):
    """Verify manual computation matches sklearn predict."""
    test_cases = [
        [0, 33, 133, 252, 211, 0, 0.5],   # blue + yellow
        [255, 0, 0, 255, 255, 255, 0.5],   # red + white
        [0, 0, 255, 255, 255, 0, 0.25],    # blue + yellow t=0.25
        [128, 64, 32, 32, 64, 128, 0.75],  # brown-ish pair
    ]

    print("Verification (sklearn vs manual):")
    max_err = 0.0
    for x in test_cases:
        x_arr = np.array([x])
        sklearn_pred = model.predict(x_arr)[0]

        feats = compute_features_manual(x, powers)
        manual_pred = feats @ coef + intercept

        err = np.max(np.abs(sklearn_pred - manual_pred))
        max_err = max(max_err, err)
        sklearn_rgb = np.clip(sklearn_pred, 0, 255).astype(int)
        manual_rgb = np.clip(manual_pred, 0, 255).astype(int)
        print(f"  Input: {x}")
        print(f"    sklearn: {list(sklearn_rgb)}")
        print(f"    manual:  {list(manual_rgb)}")
        print(f"    max abs error: {err:.2e}")

    print(f"\n  Overall max error: {max_err:.2e}")
    assert max_err < 1e-6, f"Verification failed! Max error {max_err}"
    print("  PASSED\n")


def format_c_array(name, arr, indent="    "):
    """Format a numpy array as a C array initializer."""
    lines = []
    if arr.ndim == 1:
        lines.append(f"static const double {name}[{len(arr)}] = {{")
        for i, v in enumerate(arr):
            comma = "," if i < len(arr) - 1 else ""
            lines.append(f"{indent}{v:.17e}{comma}")
        lines.append("};")
    elif arr.ndim == 2:
        rows, cols = arr.shape
        lines.append(f"static const double {name}[{rows}][{cols}] = {{")
        for i in range(rows):
            vals = ", ".join(f"{v:.17e}" for v in arr[i])
            comma = "," if i < rows - 1 else ""
            lines.append(f"{indent}{{{vals}}}{comma}")
        lines.append("};")
    return "\n".join(lines)


def format_c_int_array(name, arr, indent="    "):
    """Format a numpy integer array as a C array initializer."""
    lines = []
    rows, cols = arr.shape
    lines.append(f"static const int {name}[{rows}][{cols}] = {{")
    for i in range(rows):
        vals = ", ".join(str(int(v)) for v in arr[i])
        comma = "," if i < rows - 1 else ""
        lines.append(f"{indent}{{{vals}}}{comma}")
    lines.append("};")
    return "\n".join(lines)


def export_test_cases(model):
    """Generate test cases for C++ verification."""
    test_inputs = [
        ([0, 33, 133], [252, 211, 0], 0.5),
        ([255, 0, 0], [255, 255, 255], 0.5),
        ([0, 0, 255], [255, 255, 0], 0.25),
        ([255, 0, 0], [0, 0, 255], 0.5),
        ([128, 128, 128], [128, 128, 128], 0.5),
    ]
    lines = ["// Test cases: {r1,g1,b1, r2,g2,b2, t, expected_r, expected_g, expected_b}"]
    lines.append(f"static const int test_cases[][10] = {{")
    for c1, c2, t in test_inputs:
        x = np.array([c1 + c2 + [t]])
        pred = np.clip(model.predict(x)[0], 0, 255).astype(int)
        vals = c1 + c2 + [int(t * 1000)]  # store t*1000 as int
        vals_str = ", ".join(str(v) for v in (c1 + c2))
        lines.append(f"    {{{vals_str}, {int(pred[0])}, {int(pred[1])}, {int(pred[2])}}},")
    lines.append("};")
    # Also print for reference
    print("Test cases for C++:")
    for c1, c2, t in test_inputs:
        x = np.array([c1 + c2 + [t]])
        pred = np.clip(model.predict(x)[0], 0, 255).astype(int)
        print(f"  lerp({c1}, {c2}, {t}) -> ({pred[0]}, {pred[1]}, {pred[2]})")
    return "\n".join(lines)


def main():
    model = load_model()

    poly_features = model[0]  # PolynomialFeatures
    lin_reg = model[1]        # LinearRegression

    powers = poly_features.powers_   # shape (n_features, 7)
    coef = lin_reg.coef_             # shape (3, n_features)
    intercept = lin_reg.intercept_   # shape (3,)

    n_features = powers.shape[0]
    n_inputs = powers.shape[1]
    degree = poly_features.degree

    print(f"Model: PolynomialFeatures(degree={degree}) + LinearRegression")
    print(f"  Input dimensions: {n_inputs}")
    print(f"  Polynomial features: {n_features}")
    print(f"  Coefficient matrix: {coef.shape}")
    print(f"  Intercept: {intercept}")
    print()

    # Verify manual computation matches sklearn
    verify(model, powers, coef.T, intercept)

    # Export test cases
    export_test_cases(model)
    print()

    # Write coefficients directly into filament_mixer.h
    header_path = Path("cpp/filament_mixer.h")

    header = header_path.read_text()

    # Build the data block
    data_lines = []
    data_lines.append("// Auto-generated by scripts/export_poly_coefficients.py")
    data_lines.append("// Do not edit manually.")
    data_lines.append(f"// Degree-{degree} polynomial, {n_features} features, {n_inputs} inputs")
    data_lines.append("")
    data_lines.append(f"static const int POLY_DEGREE = {degree};")
    data_lines.append(f"static const int N_FEATURES = {n_features};")
    data_lines.append(f"static const int N_INPUTS = {n_inputs};")
    data_lines.append("")
    data_lines.append(format_c_int_array("POWERS", powers))
    data_lines.append("")
    data_lines.append(format_c_array("COEF", coef.T))
    data_lines.append("")
    data_lines.append(format_c_array("INTERCEPT", intercept))
    new_data = "\n".join(data_lines)

    # Replace everything between the markers
    BEGIN = "// BEGIN AUTO-GENERATED COEFFICIENTS"
    END = "// END AUTO-GENERATED COEFFICIENTS"

    if BEGIN in header and END in header:
        before = header[:header.index(BEGIN) + len(BEGIN)]
        after = header[header.index(END):]
        header = before + "\n" + new_data + "\n" + after
    else:
        # First time: insert markers around existing data block
        # Find the existing auto-generated comment and the compute_poly_features function
        old_start = "// Auto-generated by scripts/export_poly_coefficients.py"
        func_start = "inline void compute_poly_features"
        if old_start in header and func_start in header:
            before_data = header[:header.index(old_start)]
            after_data = header[header.index(func_start):]
            header = before_data + BEGIN + "\n" + new_data + "\n" + END + "\n\n" + after_data
        else:
            raise RuntimeError("Could not find insertion point in filament_mixer.h")

    header_path.write_text(header)
    print(f"Written: {header_path} ({header_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
