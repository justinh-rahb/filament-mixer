#!/usr/bin/env python3
"""
Automated Pigment Optimization Script

This script uses scipy.optimize to finding the optimal Gaussian parameters
(center, width, height) for FilamentMixer pigments to match Mixbox's behavior.

It minimizes the mean Delta-E between FilamentMixer results and Mixbox ground truth
across a random set of color mixes.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import mixbox
import argparse
import time
import json
from pathlib import Path

from filament_mixer import FilamentMixer, CIE_WAVELENGTHS
from filament_mixer.km_core import Pigment

def gaussian_peak(wavelengths, center, width, height):
    """Create a Gaussian-shaped spectral peak."""
    return height * np.exp(-((wavelengths - center) / width) ** 2)

# ---------------------------------------------------------------------------
# Training Data Generation
# ---------------------------------------------------------------------------

def generate_training_data(n_samples=500):
    """Generate random color pairs and their Mixbox ground truth."""
    print(f"Generating {n_samples} training samples...")
    data = []
    
    # Add specific corner cases manually
    corner_cases = [
        ((0, 33, 133), (252, 211, 0)),   # Blue + Yellow
        ((255, 39, 2), (0, 33, 133)),    # Red + Blue
        ((255, 39, 2), (252, 211, 0)),   # Red + Yellow
        ((128, 2, 46), (252, 211, 0)),   # Magenta + Yellow
        ((0, 60, 50), (128, 2, 46)),     # Green + Magenta
    ]
    
    for c1, c2 in corner_cases:
        target = mixbox.lerp(c1, c2, 0.5)
        data.append({
            "c1": c1, "c2": c2, "t": 0.5, "target": target, "weight": 5.0 # Higher weight for key colors
        })
        
    # Random samples
    np.random.seed(42)
    for _ in range(n_samples):
        c1 = tuple(np.random.randint(0, 256, 3))
        c2 = tuple(np.random.randint(0, 256, 3))
        t = np.random.uniform(0.2, 0.8)
        
        target = mixbox.lerp(c1, c2, t)
        data.append({
            "c1": c1, "c2": c2, "t": t, "target": target, "weight": 1.0
        })
        
    return data

# ---------------------------------------------------------------------------
# Parametric Pigment Factory
# ---------------------------------------------------------------------------

def make_cyan(params):
    # Params: [r_center, r_width, r_height, o_center, o_width, o_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2]) # Red peak
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5]) # Orange peak
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Cyan", K, S)

def make_magenta(params):
    # Params: [lg_center, lg_width, lg_height, ug_center, ug_width, ug_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.03
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2]) # Lower green
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5]) # Upper green
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Magenta", K, S)

def make_yellow(params):
    # Params: [b_center, b_width, b_height, v_center, v_width, v_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2]) # Blue peak
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5]) # Violet peak
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Yellow", K, S)

def make_white(params):
    # Params: [k_base, s_base]
    K = np.ones(len(CIE_WAVELENGTHS)) * params[0]
    S = np.ones(len(CIE_WAVELENGTHS)) * params[1]
    return Pigment("White", K, S)

# Initial/Reference Parameters (from pigments.py)
# Cyan:    640, 45, 2.5,  600, 20, 1.5
# Magenta: 530, 20, 3.0,  555, 20, 2.5
# Yellow:  450, 45, 2.5,  400, 40, 2.0
# White:   0.008, 1.2

INITIAL_PARAMS = np.array([
    # Cyan (6 params)
    640, 45, 2.5,  600, 20, 1.5,
    # Magenta (6 params)
    530, 20, 3.0,  555, 20, 2.5,
    # Yellow (6 params)
    450, 45, 2.5,  400, 40, 2.0,
    # White (2 params)
    0.008, 1.2
])

# Bounds for optimization
BOUNDS = [
    # Cyan
    (600, 680), (10, 100), (0.1, 5.0),  # Red peak
    (580, 620), (10, 60),  (0.1, 4.0),  # Orange peak
    # Magenta
    (500, 540), (10, 60),  (0.1, 5.0),  # Lower green
    (540, 580), (10, 60),  (0.1, 5.0),  # Upper green
    # Yellow
    (420, 480), (10, 100), (0.1, 5.0),  # Blue peak
    (380, 420), (10, 100), (0.1, 4.0),  # Violet peak
    # White
    (0.001, 0.1), (0.5, 5.0)            # Base K, Base S
]

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

# Use local import for delta_e to avoid circular dep issues if compare isn't in path
def delta_e_simple(rgb1, rgb2):
    return np.sqrt(np.sum((np.array(rgb1) - np.array(rgb2))**2)) # Simple Euclidean for speed during opt?
    # No, optimize on true Delta-E 2000 or 76? compare.py uses CIE76 via lab conversion.
    # Let's implement CIE76 quickly here for accuracy.

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def rgb_to_lab(rgb):
    # Vectorized RGB to Lab
    rgb = np.array(rgb) / 255.0
    rl, gl, bl = srgb_to_linear(rgb[0]), srgb_to_linear(rgb[1]), srgb_to_linear(rgb[2])
    
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    def f(t):
        return np.where(t > 0.008856, t ** (1/3), 7.787 * t + 16/116)
        
    L = 116 * f(Y / Yn) - 16
    a = 500 * (f(X / Xn) - f(Y / Yn))
    b = 200 * (f(Y / Yn) - f(Z / Zn))
    
    return L, a, b

def loss_function(params, training_data):
    # 1. Reconstruct palette from params
    c_params = params[0:6]
    m_params = params[6:12]
    y_params = params[12:18]
    w_params = params[18:20]
    
    palette = [
        make_cyan(c_params),
        make_magenta(m_params),
        make_yellow(y_params),
        make_white(w_params)
    ]
    
    mixer = FilamentMixer(palette)
    total_error = 0.0
    total_weight = 0.0
    
    # 2. Compute error batch
    # Optimization: This is slow because we init KM every time.
    # Ideally should vectorized, but mix() is complex. 
    # Just looping for now, standard minimization can handle it.
    
    for sample in training_data:
        c1, c2, t = sample['c1'], sample['c2'], sample['t']
        target = sample['target']
        weight = sample['weight']
        
        # FilamentMixer prediction
        predicted = mixer.lerp(*c1, *c2, t)
        
        # Loss (Weighted CIE76 Delta-E)
        # Note: rgb_to_lab handles loose inputs, but let's ensure tuple
        L1, a1, b1 = rgb_to_lab(predicted)
        L2, a2, b2 = rgb_to_lab(target)
        
        dE = np.sqrt((L1 - L2)**2 + (a1 - a2)**2 + (b1 - b2)**2)
        total_error += dE * weight
        total_weight += weight
        
    return total_error / total_weight

# ---------------------------------------------------------------------------
# Main Optimization Loop
# ---------------------------------------------------------------------------

# Global iteration counter
iteration_count = 0
training_data = None

def print_progress(xk):
    global iteration_count, training_data
    iteration_count += 1
    
    # Calculate current loss (adds ~3s overhead but worth it for visibility)
    current_loss = loss_function(xk, training_data)
    print(f"Iteration {iteration_count}: Loss = {current_loss:.4f}")

def main():
    global training_data
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)  # Reduced from 100 for speed
    parser.add_argument("--maxiter", type=int, default=50)
    args = parser.parse_args()
    
    print("Initializing Automated Optimization...")
    training_data = generate_training_data(args.samples)
    
    start_time = time.time()
    initial_loss = loss_function(INITIAL_PARAMS, training_data)
    print(f"Initial Mean Loss (Delta-E): {initial_loss:.4f}")
    
    print(f"\nStarting optimization (maxiter={args.maxiter})...")
    
    # Strategy: Run local optimization from current "good" manual parameters
    # Powell is too slow (N^2 evaluations). L-BFGS-B is O(N) with gradients.
    result = minimize(
        loss_function,
        INITIAL_PARAMS,
        args=(training_data,),
        method='L-BFGS-B',
        bounds=BOUNDS,
        callback=print_progress,
        options={'maxiter': args.maxiter, 'disp': True, 'eps': 1e-4}
    )
    
    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.1f}s")
    print(f"Final Success: {result.success}")
    print(f"Final Mean Loss (Delta-E): {result.fun:.4f}")
    print(f"Improvement: {initial_loss - result.fun:.4f}")
    
    # Report Params
    p = result.x
    print("\nOPTIMIZED PARAMETERS:")
    print("-" * 40)
    print(f"Cyan Red Peak:    Center={p[0]:.1f}, Width={p[1]:.1f}, Height={p[2]:.2f}")
    print(f"Cyan Orange Peak: Center={p[3]:.1f}, Width={p[4]:.1f}, Height={p[5]:.2f}")
    print("-" * 40)
    print(f"Magenta L-Grn Pk: Center={p[6]:.1f}, Width={p[7]:.1f}, Height={p[8]:.2f}")
    print(f"Magenta U-Grn Pk: Center={p[9]:.1f}, Width={p[10]:.1f}, Height={p[11]:.2f}")
    print("-" * 40)
    print(f"Yellow Blue Peak: Center={p[12]:.1f}, Width={p[13]:.1f}, Height={p[14]:.2f}")
    print(f"Yellow Viol Peak: Center={p[15]:.1f}, Width={p[16]:.1f}, Height={p[17]:.2f}")
    print("-" * 40)
    print(f"White Base:       K={p[18]:.4f}, S={p[19]:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
