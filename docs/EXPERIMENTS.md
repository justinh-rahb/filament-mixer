# Research Experiments: Alternative Mixing Algorithms

We are exploring three distinct approaches to improve color mixing accuracy (specifically for Green) and runtime performance.

## Experiment A: Polynomial Direct Mixing (7D → 3D) — [PRODUCTIONIZED]

**Hypothesis**: A simple polynomial regression can learn the mapping from two input colors `(R1, G1, B1, R2, G2, B2)` and the mixing ratio `t` to a mixed color `(R_mix, G_mix, B_mix)` with sufficient accuracy to replace the physics engine entirely.

-   **Model**: `sklearn.preprocessing.PolynomialFeatures(degree=4)` + `LinearRegression`.
-   **Training Data**: 150,000 random pairs mixed with `mixbox` (ground truth), using continuous uniform `t` sampling.
-   **Pros**:
    -   **Instant Inference**: Just matrix multiplication (~0.001ms).
    -   **Accurate**: Captures subtle gamut nuances better than degree-3.
    -   **Perfect Endpoints**: Hard-clamped in code to ensure 100% accuracy at $t=0$ and $t=1$.
-   **Cons**:
    -   **Black Box**: Doesn't "understand" pigments; relies on training data coverage.
    -   **Pairwise Only**: Mixing 3+ colors requires iterative mixing.

**Implementation**: `src/filament_mixer/poly_mixer.py`

### Results (2026-02-11)
- **Mean Delta-E**: **2.07** (vs 11.77 for Physics Engine).
- **Speed**: **0.001ms** per mix (Instant).
- **Blue + Yellow Test**:
    - Predicted (t=0.5): `[47, 141, 56]` (Vibrant Green)
    - Mixbox Ref: `[41, 130, 57]`
    - **Result**: Successfully captured the "green" mixing behavior with excellent perceptual match.

**Conclusion**: The 4th-degree polynomial is **the chosen production model** for high-performance mixing. It provides the best balance of speed, simplicity, and accuracy.


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


---

## Experiment E: Wondermixer (Pre-fitted Total Degree 3 Polynomial)

**Hypothesis**: A pre-trained cubic polynomial with 120 coefficients fitted to MIXBOX data can provide excellent color mixing accuracy with minimal dependencies (only numpy), offering a lightweight alternative that doesn't require training infrastructure.

-   **Model**: Full monomial feature map up to total degree 3 for 7D input `[(R1, G1, B1), (R2, G2, B2), T]`.
-   **Features**: 120 polynomial terms = C(7+3, 3) via canonical ordering (degree 0, 1, 2, 3).
-   **Training Data**: Pre-fitted on MIXBOX ground truth by WombleyRole (Snapmaker U1).
-   **Input Transform**: RGB values normalized to [0,1], then mapped to [-1,1] before feature expansion.
-   **Pros**:
    -   **Zero Training**: Pre-fitted coefficients included (no sklearn/scipy needed).
    -   **Minimal Dependencies**: Only requires numpy.
    -   **Fast Inference**: Matrix multiplication only (~0.001ms).
    -   **Good Mid-Range**: Excellent accuracy for mixing ratios 0.2-0.8.
-   **Cons**:
    -   **Extreme Ratio Issues**: Can be notably off at very low (<0.1) or very high (>0.9) mixing ratios, especially with white.
    -   **Pairwise Only**: Like polynomial/GP approaches, mixing 3+ colors requires iterative mixing.
    -   **Black Box**: Pre-trained weights without visibility into hypercubic fit process.

**Implementation**: `scripts/experiment_wondermixer.py`

**Credit**: Wombley
**Approximate fit to**: MIXBOX by Sarka Sochorova and Ondrej Jamriska  
  - Website: https://scrtwpns.com/mixbox/  
  - GitHub: https://github.com/scrtwpns/mixbox

### Results (2026-02-11)
- **Mean Delta-E**: **3.38** (Better than Logit Poly's 5.43, but not as good as Poly's 2.07).
- **Speed**: **0.1045 ms/sample** (Slower than Poly's 0.001ms, but still quite fast).
- **Delta-E by Mixing Ratio**:
    - **Extreme ratios** (<0.2 or >0.8): **3.08** (Good!)
    - **Mid-range** (0.35-0.65): **4.08** (Surprisingly worse than extremes)
    - **Other ratios**: **3.07** (Good)
- **Edge Case Testing**:
    - **t=0.0 (100% red)**: `[255, 0, 0]` ✓ Perfect!
    - **t=1.0 (100% white)**: `[255, 218, 249]` ✗ Pink tint (expected `[255, 255, 255]`)
    - **t=0.05 (Red+White)**: ΔE=1.62 ✓ Excellent
    - **Yellow+Blue (t=0.5)**: `[80, 158, 92]` vs Mixbox `[78, 150, 100]` - Good green

**Known Issues** (confirmed):
- **Endpoint problem at t=1.0**: When mixing toward the second color (t→1), some color pairs produce incorrect tints. Red→White gives pink instead of pure white.
- **Mid-range accuracy**: Surprisingly, the mid-range (0.35-0.65) has slightly higher error than extreme ratios, contrary to initial expectations.
- **Not as good as Poly**: The 4th-degree polynomial approach (Experiment A) significantly outperforms this 3rd-degree approach (dE 2.07 vs 3.38).

**Conclusion**: The Wondermixer provides a **lightweight, numpy-only** alternative with reasonable accuracy (dE 3.38), but doesn't match the production PolyMixer (dE 2.07). The pre-fitted coefficients eliminate training time, making it attractive for scenarios where sklearn is unavailable. However, the t=1.0 endpoint issue and mid-range performance gaps suggest the 4th-degree polynomial with proper training data coverage (Experiment A) remains superior for production use. Could be useful as a fallback or for embedded systems with limited dependencies.


