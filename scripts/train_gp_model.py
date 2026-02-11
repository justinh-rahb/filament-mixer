#!/usr/bin/env python3
"""
Train production Gaussian Process model for Experiment C.

This trains a GP regressor on Mixbox ground truth data for direct
RGB₁ + RGB₂ + t → RGB_mix mapping. No concentration space or unmixing needed.

The model learns to mimic Mixbox's behavior with high accuracy while being
much faster than the physics engine.

Usage:
    python scripts/train_gp_model.py
"""

import numpy as np
import mixbox
import pickle
import time
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from skimage import color


def generate_training_data(n_samples: int):
    """
    Generate training data from Mixbox ground truth.
    
    Args:
        n_samples: Number of random color pair samples to generate
    
    Returns:
        X: Input features (n_samples, 7) - [r1, g1, b1, r2, g2, b2, t] with RGB in [0, 1], t in [0, 1]
        y: Target outputs (n_samples, 3) - [r_mix, g_mix, b_mix] scaled to [0, 1]
    """
    print(f"Generating {n_samples:,} training samples from Mixbox...")
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random color pairs
        c1 = tuple(np.random.randint(0, 256, 3))
        c2 = tuple(np.random.randint(0, 256, 3))
        # Random mixing ratio
        t = np.random.random()
        
        # Ground truth from Mixbox
        mixed = mixbox.lerp(c1, c2, t)
        
        # Store: RGB scaled to [0,1], t already in [0,1]
        X.append([c1[0]/255., c1[1]/255., c1[2]/255., 
                  c2[0]/255., c2[1]/255., c2[2]/255., t])
        y.append(mixed)
    
    X = np.array(X, dtype=np.float32)  # Already scaled properly
    y = np.array(y, dtype=np.float32) / 255.0  # Scale to [0, 1]
    
    return X, y


def evaluate_accuracy(model, n_test: int = 500):
    """
    Evaluate model accuracy on test set.
    
    Args:
        model: Trained GP model
        n_test: Number of test samples
    
    Returns:
        Mean Delta-E CIE76 score
    """
    print(f"\nEvaluating on {n_test:,} test samples...")
    
    # Generate test data
    X_test, y_test = generate_training_data(n_test)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)
    
    # Compute Delta-E for each sample
    deltas = []
    for i in range(n_test):
        # rgb2lab expects values in [0, 1] range
        rgb_true = y_test[i].reshape(1, 1, 3)
        rgb_pred = y_pred[i].reshape(1, 1, 3)
        
        lab_true = color.rgb2lab(rgb_true)
        lab_pred = color.rgb2lab(rgb_pred)
        
        dE = np.sqrt(np.sum((lab_true - lab_pred)**2))
        deltas.append(dE)
    
    mean_dE = np.mean(deltas)
    max_dE = np.max(deltas)
    
    print(f"  Mean Delta-E: {mean_dE:.2f}")
    print(f"  Max Delta-E:  {max_dE:.2f}")
    
    return mean_dE


def test_blue_yellow(model):
    """Test the critical Blue + Yellow → Green case."""
    print("\n--- Blue + Yellow Test ---")
    
    c1 = (0, 33, 133)   # Blue
    c2 = (252, 211, 0)  # Yellow
    t = 0.5
    
    # Prepare input
    X = np.array([[c1[0]/255., c1[1]/255., c1[2]/255.,
                   c2[0]/255., c2[1]/255., c2[2]/255., t]])
    
    # Predict
    rgb_pred = model.predict(X)[0] * 255.0
    rgb_pred = np.clip(rgb_pred, 0, 255).astype(int)
    
    # Ground truth
    rgb_truth = mixbox.lerp(c1, c2, t)
    
    print(f"  Input 1 (Blue):   {c1}")
    print(f"  Input 2 (Yellow): {c2}")
    print(f"  Predicted:        {tuple(rgb_pred)}")
    print(f"  Mixbox Truth:     {rgb_truth}")
    
    # Delta-E (rgb2lab expects [0, 1] range)
    lab_pred = color.rgb2lab((rgb_pred / 255).reshape(1, 1, 3))
    lab_truth = color.rgb2lab((np.array(rgb_truth) / 255).reshape(1, 1, 3))
    dE = np.sqrt(np.sum((lab_pred - lab_truth)**2))
    print(f"  Delta-E:          {dE:.2f}")


def main():
    """Train and save the GP model."""
    
    # Configuration
    N_TRAIN = 2000  # More samples = better accuracy, but slower training (O(N³))
    OUTPUT_DIR = Path("models")
    MODEL_FILE = "gp_model.pkl"
    
    print("=" * 60)
    print("Training Gaussian Process Model (Experiment C)")
    print("=" * 60)
    
    # 1. Generate training data
    X_train, y_train = generate_training_data(N_TRAIN)
    
    # 2. Configure and train GP
    print(f"\nTraining Gaussian Process Regressor...")
    print(f"  Kernel: RBF + WhiteKernel")
    print(f"  Expected training time: ~10-20s for {N_TRAIN:,} samples")
    
    # RBF kernel with length scale tuned for [0, 1] normalized space
    kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1.0)) + \
             WhiteKernel(noise_level=1e-5)
    
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,  # Use default initialization for speed
        alpha=1e-10  # Regularization
    )
    
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0
    
    print(f"✓ Training complete in {training_time:.1f}s")
    print(f"  Optimized kernel: {model.kernel_}")
    
    # 3. Evaluate accuracy
    mean_dE = evaluate_accuracy(model, n_test=1000)
    
    # 4. Test critical case
    test_blue_yellow(model)
    
    # 5. Benchmark inference speed
    print("\n--- Speed Benchmark ---")
    n_speed_test = 1000
    X_speed = np.random.random((n_speed_test, 7))
    
    t0 = time.time()
    _ = model.predict(X_speed)
    duration = time.time() - t0
    
    ms_per_mix = (duration / n_speed_test) * 1000
    print(f"  Inference speed: {ms_per_mix:.4f} ms/mix")
    
    # 6. Save model
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / MODEL_FILE
    
    print(f"\nSaving model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n{'='*60}")
    print("✓ GP Model Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_path}")
    print(f"Training samples: {N_TRAIN:,}")
    print(f"Mean Delta-E: {mean_dE:.2f}")
    print(f"Inference speed: {ms_per_mix:.4f} ms/mix")
    print(f"\nTo use in code:")
    print(f"  from filament_mixer.gp_mixer import GPMixer")
    print(f"  mixer = GPMixer()")
    print(f"  result = mixer.lerp(r1, g1, b1, r2, g2, b2, t)")
    

if __name__ == "__main__":
    main()
