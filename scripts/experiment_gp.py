
import numpy as np
import time
import mixbox
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from skimage import color

def generate_data(n_samples=2000):
    print(f"Generating {n_samples} training samples (Ground Truth: Mixbox)...")
    X = []
    y = []
    for _ in range(n_samples):
        c1 = list(np.random.randint(0, 256, 3))
        c2 = list(np.random.randint(0, 256, 3))
        mixed = mixbox.lerp(tuple(c1), tuple(c2), 0.5)
        X.append(c1 + c2)
        y.append(mixed)
    return np.array(X), np.array(y)

def evaluate_model(model, n_test=500):
    print(f"Evaluating on {n_test} test samples...")
    X_test, y_test = generate_data(n_test)
    
    start_time = time.time()
    y_pred, sigma = model.predict(X_test, return_std=True)
    duration = time.time() - start_time
    
    y_pred = np.clip(y_pred, 0, 255)
    
    total_dE = 0
    for i in range(n_test):
        rgb_true = y_test[i].reshape(1, 1, 3) / 255.0
        rgb_pred = y_pred[i].reshape(1, 1, 3) / 255.0
        lab_true = color.rgb2lab(rgb_true)
        lab_pred = color.rgb2lab(rgb_pred)
        dE = np.sqrt(np.sum((lab_true - lab_pred)**2))
        total_dE += dE
        
    mean_dE = total_dE / n_test
    print(f"\nResults:")
    print(f"  Mean Delta-E: {mean_dE:.4f}")
    print(f"  Inference Time: {duration*1000/n_test:.4f} ms/sample")
    print(f"  Mean Uncertainty (std): {np.mean(sigma):.4f}")
    
    return mean_dE

def main():
    # 1. Generate Data (Smaller N due to O(N^3) complexity)
    X_train, y_train = generate_data(1000) 
    
    # 2. Train Model
    print("\nTraining Gaussian Process (RBF Kernel)...")
    kernel = 1.0 * RBF(length_scale=100.0) + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Training complete in {time.time() - t0:.2f}s")
    print(f"Learned Kernel: {model.kernel_}")
    
    # 3. Evaluate
    evaluate_model(model)
    
    # 4. Test: Blue + Yellow
    c1 = [0, 33, 133]
    c2 = [252, 211, 0]
    X_custom = np.array([c1 + c2])
    pred, std = model.predict(X_custom, return_std=True)
    pred_int = np.clip(pred.astype(int), 0, 255)
    truth = mixbox.lerp(tuple(c1), tuple(c2), 0.5)
    
    print("\nTest Case: Blue + Yellow")
    print(f"  Input 1: {c1}")
    print(f"  Input 2: {c2}")
    print(f"  Prediction: {list(pred_int[0])} (Uncertainty: {std})")
    print(f"  Mixbox Ref: {list(truth)}")

if __name__ == "__main__":
    main()
