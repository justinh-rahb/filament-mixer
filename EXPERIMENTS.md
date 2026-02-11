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
