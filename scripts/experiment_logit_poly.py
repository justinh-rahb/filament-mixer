
import numpy as np
import time
import mixbox
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from skimage import color
from scipy.special import logit, expit

EPSILON = 1e-5

def generate_data(n_samples=50000):
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

def evaluate_model(model, n_test=1000):
    print(f"Evaluating on {n_test} test samples...")
    X_test, y_test = generate_data(n_test)
    
    start_time = time.time()
    
    # Predict in Logit space
    z_pred = model.predict(X_test)
    
    # Transform back to RGB [0, 255]
    y_pred = expit(z_pred) * 255.0
    
    duration = time.time() - start_time
    
    y_pred = np.clip(y_pred, 0, 255)
    
    total_dE = 0
    for i in range(n_test):
        # Convert RGB to Lab (skimage uses 0..1)
        rgb_true = y_test[i].reshape(1, 1, 3) / 255.0
        rgb_pred = y_pred[i].reshape(1, 1, 3) / 255.0
        lab_true = color.rgb2lab(rgb_true)
        lab_pred = color.rgb2lab(rgb_pred)
        dE = np.sqrt(np.sum((lab_true - lab_pred)**2))
        total_dE += dE
        
    mean_dE = total_dE / n_test
    print(f"\nResults (Exp D - Logit Poly):")
    print(f"  Mean Delta-E: {mean_dE:.4f}")
    print(f"  Inference Time: {duration*1000/n_test:.4f} ms/sample")
    return mean_dE

def main():
    # 1. Generate Data
    X_train, y_train = generate_data(20000) # Medium dataset size
    
    # 2. Transform Targets: [0, 255] -> Logit Space
    # Normalize to [0, 1]
    y_norm = y_train / 255.0
    # Clip to avoid infinity at 0 or 1
    y_clamped = np.clip(y_norm, EPSILON, 1.0 - EPSILON)
    # Logit transform
    z_train = logit(y_clamped)
    
    # 3. Train Model on Logits
    print("\nTraining Logit-Polynomial Regression (Degree=3)...")
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    
    t0 = time.time()
    model.fit(X_train, z_train)
    print(f"Training complete in {time.time() - t0:.2f}s")
    
    # 4. Evaluate
    evaluate_model(model)
    
    # 5. Test: Blue + Yellow
    c1 = [0, 33, 133]
    c2 = [252, 211, 0]
    X_custom = np.array([c1 + c2])
    
    z_pred = model.predict(X_custom)
    y_pred = expit(z_pred) * 255.0
    pred_int = y_pred.astype(int)
    
    truth = mixbox.lerp(tuple(c1), tuple(c2), 0.5)
    
    print("\nTest Case: Blue + Yellow")
    print(f"  Input 1: {c1}")
    print(f"  Input 2: {c2}")
    print(f"  Prediction: {list(pred_int[0])}")
    print(f"  Mixbox Ref: {list(truth)}")

if __name__ == "__main__":
    main()
