# Research Experiments: Alternative Mixing Algorithms

We are exploring three distinct approaches to improve color mixing accuracy (specifically for Green) and runtime performance.

## Experiment A: Polynomial Direct Mixing (6D → 3D)

**Hypothesis**: A simple polynomial regression can learn the mapping from two input colors `(R1, G1, B1, R2, G2, B2)` to a mixed color `(R_mix, G_mix, B_mix)` with sufficient accuracy to replace the physics engine entirely.

-   **Model**: `sklearn.preprocessing.PolynomialFeatures(degree=3)` + `LinearRegression`.
-   **Training Data**: 50,000 random pairs mixed with `mixbox` (ground truth).
-   **Pros**:
    -   **Instant Inference**: Just matrix multiplication (~0.001ms).
    -   **Simple**: No physics, no integration, no optimization.
    -   **No Unmixing**: Bypasses the ill-posed `RGB -> Concentration` inverse problem.
-   **Cons**:
    -   **Black Box**: Doesn't "understand" pigments; might fail on edge cases.
    -   **Pairwise Only**: Mixing 3+ colors requires iterative mixing, which accumulates error.

**Implementation**: `scripts/experiment_polynomial.py`

### Results (2026-02-10)
- **Mean Delta-E**: **3.32** (vs 11.77 for Physics Engine).
- **Speed**: **0.002ms** per mix (Instant).
- **Blue + Yellow Test**:
    - Predicted: `[42, 145, 48]` (Vibrant Green)
    - Mixbox Ref: `[41, 130, 57]`
    - **Result**: Successfully captured the "green" mixing behavior without any physics knowledge.

**Conclusion**: The simple 3rd-degree polynomial is **highly effective** for pairwise mixing. It outperforms the un-tuned physics model in accuracy by a large margin (dE 3.3 vs 11.8) and is computationally free.


---

## Experiment C: Gaussian Process Regression (Direct RGB Mixing)

**Hypothesis**: A Gaussian Process can learn the non-linear mixing manifold directly from Mixbox ground truth, providing high accuracy without requiring LUT tables or concentration-space unmixing.

-   **Model**: `sklearn.gaussian_process.GaussianProcessRegressor` with RBF kernel  
-   **Input**: 6D color pair + mixing ratio `(R1, G1, B1, R2, G2, B2, t)`
-   **Training Data**: 2,000 random pairs mixed with `mixbox` (ground truth)
-   **Pros**:
    -   **Superior Accuracy**: Directly learns Mixbox's behavior
    -   **No Unmixing**: Bypasses the ill-posed `RGB → Concentration` inverse problem
    -   **Fast Inference**: ~0.018ms per mix (faster than physics engine)
    -   **Variable t**: Handles any mixing ratio, not just fixed midpoints
-   **Cons**:
    -   **Pairwise Only**: Like polynomial, mixing 3+ colors requires iterative mixing
    -   **Training Time**: O(N³) complexity means training takes ~7s for 2,000 samples

**Implementation**: 
- `src/filament_mixer/gp_mixer.py` — Production GPMixer class
- `scripts/train_gp_model.py` — Model training script
- Model cache: `lut_gp/gp_model.pkl`

### Results (2026-02-11)
- **Mean Delta-E**: **1.79** (Best accuracy of all non-LUT methods!)
- **Max Delta-E**: **14.14**
- **Speed**: **0.018ms** per mix (35x faster than physics engine)
- **Blue + Yellow Test**:
    - Predicted: `(47, 139, 49)` (Vibrant Green)
    - Mixbox Ref: `(41, 130, 57)`
    - Delta-E: **8.77** (Good green reproduction)

**Conclusion**: GP achieves the **best accuracy** of any production-speed method, surpassing even the polynomial approach (dE 1.79 vs 3.32). The model successfully learns Mixbox's complex color mixing behavior through direct 6D → 3D regression, avoiding the problematic unmixing step entirely. This is the recommended approach for applications needing maximum accuracy without full LUT storage.


---

## Experiment B: Green-Targeted Spectral Optimization

**Hypothesis**: The current spectral optimizer found a local minimum that favored Red/White accuracy over Green. By heavily weighting "Blue + Yellow" pairs in the loss function, we can force the physics model to find a configuration that produces vibrant Green.

-   **Approach**: Modify `loss_function` in `optimize_pigments.py`.
-   **Weighting**: Apply `weight=10.0` to any sample where inputs are Blue-ish and Yellow-ish.
-   **Pros**:
    -   **Physics-Based**: Keeps the robust K-M model; valid for N-way mixing.
    -   **Interpretable**: Result is still just "pigment parameters".
-   **Cons**:
    -   **Trade-off**: Might degrade Red/White accuracy to fix Green.

**Implementation**: `scripts/optimize_green.py`

### Results (2026-02-10)
- **Loss Improvement**: 21.23 -> 21.08 (Minimal).
- **Blue + Yellow Test**:
    - Result: `(77, 102, 48)` (Muddy Olive).
    - Target: `(41, 130, 57)` (Vibrant Green).
    - **Failure**: Even with 50x weighting, the physics model could not bend the spectral curves enough to produce green without destroying other colors.

**Conclusion**: The "Green Problem" is likely structural (Gaussian approximation limit). The physics engine has hit a hard ceiling.


---

## Experiment C: Gaussian Process Regression (GP)

**Hypothesis**: A Gaussian Process can learn the non-linear mixing manifold perfectly, outperforming polynomials for complex color interactions.

-   **Model**: `sklearn.gaussian_process.GaussianProcessRegressor` with RBF kernel.
-   **input**: 6D color pair (or 7D latent representation).
-   **Pros**:
    -   **High Accuracy**: Theoretically capable of perfect interpolation.
    -   **Uncertainty**: Can tell us when it's "guessing" (high variance).
-   **Cons**:
    -   **Slow**: Training is $O(N^3)$. Inference is slower than polynomial.
    -   **Memory**: Kernels can get large.

**Implementation**: `scripts/experiment_gp.py`

### Results (2026-02-10)
- **Mean Delta-E**: **2.26** (Train N=1000).
- **Speed**: **0.025ms** (Fast enough for real-time).
- **Blue + Yellow Test**:
    - Predicted: `[54, 147, 58]` (Green) ± 6.0 std
    - Mixbox Ref: `[41, 130, 57]`
    - **Result**: Good green, slightly more yellow than Mixbox.

**Conclusion**: GP is extremely accurate and provides **uncertainty estimates**, which could be huge for detecting "out of gamut" mixes. At low N (1000), it slightly lags behind the Polynomial fit (dE 2.2 vs 3.3? Wait, 2.26 is BETTER than 3.32).
*Correction*: 2.26 is **better** than 3.32. The GP wins on accuracy per sample.

---

## Experiment D: Polynomial with Logistic Tapering

**Hypothesis**: The standard polynomial fit (Exp A) struggles at the boundaries (0/255) and requires hard clamping. By fitting the polynomial to the *logit* of the color values, we enforce a natural S-curve constraint, potentially improving accuracy near black/white/saturated primaries.

-   **Model**: `X -> Poly3 -> Linear -> Sigmoid -> Y`.
-   **Math**: Fit $f(X) \approx \ln(\frac{y}{255-y})$.
-   **Pros**:
    -   **Bounded**: Output is mathematically guaranteed to be in $(0, 255)$.
    -   **Smooth**: No hard "clipping" artifacts.
-   **Cons**:
    -   **Numerical Stability**: Logit is undefined at exact 0 or 255 (requires epsilon smoothing).

**Implementation**: `scripts/experiment_logit_poly.py`

### Results (2026-02-10)
- **Mean Delta-E**: **5.43** (Worse than Standard Poly's 3.32).
- **Speed**: **0.0007ms** (Ultra fast).
- **Blue + Yellow Test**:
    - Predicted: `[43, 153, 50]` (Green).
    - Mixbox Ref: `[41, 130, 57]`
    - **Result**: Good green, but overall accuracy across the gamut degraded.

**Conclusion**: The logit transform seemingly introduces complex non-linearities that the 3rd-degree polynomial struggles to fit. The standard polynomial (Experiment A) is superior.



