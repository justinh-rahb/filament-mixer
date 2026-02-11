#!/usr/bin/env python3
"""
Train Polynomial Mixer Model (Experiment A)

Trains a degree-3 polynomial regression on Mixbox ground truth data
and saves the model to lut_poly/poly_model.pkl.

Usage:
    python scripts/train_poly_model.py
    python scripts/train_poly_model.py --samples 200000 --output-dir my_poly
"""

import argparse
import pickle
import time
from pathlib import Path

import mixbox
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

try:
    from skimage import color as skcolor

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def generate_data(n_samples=200000):
    """Generate training data from Mixbox ground truth.

    Args:
        n_samples: Total number of samples to generate.

    Returns:
        X: (n_samples, 7) array of [r1, g1, b1, r2, g2, b2, t]
        y: (n_samples, 3) array of [r_mix, g_mix, b_mix]
    """
    print(f"Generating {n_samples:,} training samples using continuous t-sampling...")

    rng = np.random.default_rng(42)

    X = np.zeros((n_samples, 7))
    y = np.zeros((n_samples, 3))

    for i in range(n_samples):
        c1 = rng.integers(0, 256, 3).tolist()
        c2 = rng.integers(0, 256, 3).tolist()
        t = rng.uniform(0.0, 1.0)

        mixed = mixbox.lerp(tuple(c1), tuple(c2), t)

        X[i] = c1 + c2 + [t]
        y[i] = mixed

    return X, y


def compute_delta_e(rgb1, rgb2):
    """CIE76 Delta-E between two sRGB colors (0-255 each)."""
    if HAS_SKIMAGE:
        lab1 = skcolor.rgb2lab(np.array(rgb1).reshape(1, 1, 3) / 255.0)
        lab2 = skcolor.rgb2lab(np.array(rgb2).reshape(1, 1, 3) / 255.0)
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
    else:
        # Fallback: simple Euclidean in RGB space
        return float(np.sqrt(np.sum((np.array(rgb1) - np.array(rgb2)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Train polynomial mixer model")
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="Number of training samples (default: 200000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10000,
        help="Number of test samples (default: 10000)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=4,
        help="Polynomial degree (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lut_poly",
        help="Output directory for model pickle (default: lut_poly)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Training Polynomial Mixer (Degree={args.degree})")
    print("=" * 70)

    # 1. Generate training data
    X_train, y_train = generate_data(args.samples)

    # 2. Train model
    print(f"\nTraining PolynomialFeatures(degree={args.degree}) + LinearRegression...")
    model = make_pipeline(PolynomialFeatures(args.degree), LinearRegression())

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training complete in {train_time:.1f}s")

    n_features = model[0].n_output_features_
    print(f"  Polynomial features: {n_features}")

    # 3. Evaluate
    print(f"\nEvaluating on {args.test_samples:,} test samples...")
    X_test, y_test = generate_data(args.test_samples)

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0
    y_pred = np.clip(y_pred, 0, 255)

    # Compute Delta-E
    delta_es = []
    for i in range(args.test_samples):
        dE = compute_delta_e(y_test[i], y_pred[i])
        delta_es.append(dE)

    delta_es = np.array(delta_es)
    print(f"\n  Results:")
    print(f"  Mean Delta-E:   {delta_es.mean():.2f}")
    print(f"  Median Delta-E: {np.median(delta_es):.2f}")
    print(f"  Max Delta-E:    {delta_es.max():.2f}")
    print(f"  < 2.0 (imperceptible): {np.sum(delta_es < 2.0):,}/{args.test_samples:,}")
    print(f"  < 5.0 (minor):         {np.sum(delta_es < 5.0):,}/{args.test_samples:,}")
    print(f"  Inference: {pred_time*1000/args.test_samples:.4f} ms/sample")

    # 4. Signature test: Blue + Yellow
    print("\n  Blue + Yellow Test (The Green Problem):")
    for t in [0.25, 0.5, 0.75]:
        X_test_case = np.array([[0, 33, 133, 252, 211, 0, t]])
        pred = np.clip(model.predict(X_test_case)[0], 0, 255).astype(int)
        truth = mixbox.lerp((0, 33, 133), (252, 211, 0), t)
        dE = compute_delta_e(pred, truth)
        print(f"    t={t}: Pred={list(pred)}, Mixbox={list(truth)}, dE={dE:.2f}")

    # 5. Save model
    model_path = output_dir / "poly_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n  Model saved to: {model_path}")
    print(f"  Model size: {model_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
