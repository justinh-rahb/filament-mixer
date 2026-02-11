
import numpy as np
import time
import mixbox  # Assuming 'pymixbox' is installed as 'mixbox' or similar
# If pymixbox is not installed, we might need a mock or install it.
# Based on previous context, 'mixbox' library is available.

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from skimage import color

def lerp_rgb(c1, c2, t):
    return [int((1-t)*c1[i] + t*c2[i]) for i in range(3)]

def generate_data(n_samples=50000):
    # Generating synthetic Mixbox data 
    # Logic: For each sample, pick 2 random colors, ask Mixbox (or our grounded truth) for the mix at t=0.5
    
    # Check if mixbox is actually available
    try:
        import mixbox
    except ImportError:
        print("Error: 'mixbox' library not found. Cannot generate ground truth.")
        exit(1)

    print(f"Generating {n_samples} training samples (Ground Truth: Mixbox)...")
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # random RGB 0-255
        c1 = list(np.random.randint(0, 256, 3))
        c2 = list(np.random.randint(0, 256, 3))
        
        # Mixbox mix at t=0.5
        # The library might be `pymixbox` or similar. Let's assume standard usage.
        # Based on context: `mixbox.lerp(c1, c2, t)`
        mixed = mixbox.lerp(tuple(c1), tuple(c2), 0.5)
        
        # Input features: [r1, g1, b1, r2, g2, b2]
        X.append(c1 + c2)
        y.append(mixed)
        
    return np.array(X), np.array(y)

def evaluate_model(model, n_test=1000):
    print(f"Evaluating on {n_test} test samples...")
    X_test, y_test = generate_data(n_test)
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    duration = time.time() - start_time
    
    # Clip predictions
    y_pred = np.clip(y_pred, 0, 255)
    
    # Compute Delta-E
    total_dE = 0
    for i in range(n_test):
        # Convert RGB to Lab
        # skimage expects float 0..1
        rgb_true = y_test[i].reshape(1, 1, 3) / 255.0
        rgb_pred = y_pred[i].reshape(1, 1, 3) / 255.0
        
        lab_true = color.rgb2lab(rgb_true)
        lab_pred = color.rgb2lab(rgb_pred)
        
        # Euclidean distance in Lab
        dE = np.sqrt(np.sum((lab_true - lab_pred)**2))
        total_dE += dE
        
    mean_dE = total_dE / n_test
    print(f"\nResults:")
    print(f"  Mean Delta-E: {mean_dE:.4f}")
    print(f"  Inference Time: {duration*1000/n_test:.4f} ms/sample")
    
    return mean_dE

def main():
    # 1. Generate Training Data
    X_train, y_train = generate_data(10000) # Start small
    
    # 2. Train Model
    print("\nTraining Polynomial Regression (Degree=3)...")
    # Degree 3 allows for interaction terms like r1*r2, r1^2*g2, etc.
    # This mimics the multiplicative nature of subtractive mixing.
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Training complete in {time.time() - t0:.2f}s")
    
    # 3. Evaluate
    evaluate_model(model)
    
    # 4. Visible Test: Blue + Yellow (The "Green Problem")
    # Blue: [0, 33, 133], Yellow: [252, 211, 0]
    c1 = [0, 33, 133]
    c2 = [252, 211, 0]
    
    X_custom = np.array([c1 + c2])
    pred = model.predict(X_custom)[0]
    pred_int = pred.astype(int)
    pred_int = np.clip(pred_int, 0, 255)
    
    truth = mixbox.lerp(tuple(c1), tuple(c2), 0.5)
    
    print("\nTest Case: Blue + Yellow")
    print(f"  Input 1: {c1}")
    print(f"  Input 2: {c2}")
    print(f"  Prediction: {list(pred_int)} (Green?)")
    print(f"  Mixbox Ref: {list(truth)}")
    print("\n--- Edge Case Stress Test (Raw Predictions) ---")
    corners = [
        ([0, 0, 0], [0, 0, 0], "Black+Black"),
        ([255, 255, 255], [255, 255, 255], "White+White"),
        ([255, 0, 0], [0, 255, 0], "Red+Green"),
        ([0, 0, 255], [255, 255, 0], "Blue+Yellow"),
    ]
    
    for c1, c2, name in corners:
        X_case = np.array([c1 + c2])
        raw_pred = model.predict(X_case)[0]
        print(f"{name}: {raw_pred} -> Clamped: {np.clip(raw_pred, 0, 255).astype(int)}")

if __name__ == "__main__":
    main()
