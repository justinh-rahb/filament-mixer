# FilamentMixer: Physics-Based Color Mixing for 3D Printing

## Development Practices

**IMPORTANT: Always use `./venv/` for Python virtual environment**

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
pip install -e ".[dev]"

# Run benchmarks  (generates PNG comparison images)
python benchmarks/compare.py                    # Generates comparison.png
python benchmarks/visual_compare.py             # Generates visual_comparison.png + gradient_comparison.png
python benchmarks/compare_with_lut.py           # LUT-specific benchmarks
python benchmarks/compare_poly.py               # Poly-specific benchmarks

# Deactivate when done
deactivate
```

**LUT Resolution:** 
- Always use 256³ LUT resolution (not 64³) for production quality
- Default in visual_compare.py is 256³
- Override with `--lut-resolution 256` if needed
- 64³ is only for quick testing/development

**Benchmark Outputs:**
- `benchmarks/comparison.png` - Matplotlib chart comparing all methods
- `benchmarks/visual_comparison.png` - Swatch grid showing all mixers
- `benchmarks/gradient_comparison.png` - Gradient strips for smooth transitions

## Project Purpose

FilamentMixer implements physically-accurate color mixing for multi-material 3D printers. It replaces naive RGB interpolation (which produces muddy, desaturated blends) with spectral-based pigment physics using Kubelka-Munk theory.

**The core problem:** Traditional slicers use `(Color A + Color B) / 2` to calculate mixing ratios. This treats colors like light beams (additive), but physical pigments mix **subtractively** — they absorb and scatter light. Blue + Yellow should give vibrant Green, not gray.

**The solution:** Model each filament as a physical pigment with wavelength-dependent absorption (K) and scattering (S) properties, mix in spectral space, then convert back to RGB.

## Architecture Overview

```
src/filament_mixer/
├── km_core.py       # Kubelka-Munk physics engine (K-M equations 1-7)
├── pigments.py      # Spectral definitions for CMYW/CMYK/RYBW palettes
├── unmixer.py       # RGB → pigment concentration solver
├── api.py           # FilamentMixer class (main entry point)
├── lut.py           # Lookup table generator for fast runtime mixing
├── poly_mixer.py    # PolyMixer - Production polynomial model
└── gp_mixer.py      # GPMixer - Production Gaussian Process model

scripts/
├── optimize_pigments.py    # Differentiable spectral optimizer
├── train_poly_model.py     # Train polynomial model
├── train_gp_model.py       # Train GP model
├── generate_lut.py         # Generate physics-based LUTs
├── experiment_*.py         # Research experiments (A-E)
└── lut_demo.py            # Performance demonstrations
```

### Key Classes

- **`Pigment`** — Dataclass containing name, K spectrum (38 wavelengths), S spectrum
- **`KubelkaMunk`** — Physics engine that implements the full mixing pipeline
- **`RGBUnmixer`** — Constrained optimizer (SLSQP) that solves RGB → concentrations
- **`FilamentMixer`** — High-level API that combines physics components
- **`PolyMixer`** — Production polynomial model (degree 4, dE 2.07, 0.001ms)
- **`GPMixer`** — Production Gaussian Process model (dE 1.79, 0.018ms)

## How Color Mixing Works

### 1. Naive RGB Approach (What This Replaces)

```python
# Traditional slicer approach
result = (1 - t) * color1 + t * color2
```

**Problem:** Treats RGB channels as independent light sources. Blue (0, 0, 255) + Yellow (255, 255, 0) = (127, 127, 127) — a muddy gray, not green.

**Why it fails:** Ignores that pigments absorb complementary wavelengths. Physical mixing is a multiplication (light gets removed), not addition.

### 2. Kubelka-Munk Physics (This Implementation)

The full pipeline (Equations 1-7 from the Mixbox paper):

#### Step 1: Mix K and S Spectra (Equation 1)
```python
K_mix = c1*K1 + c2*K2 + c3*K3 + c4*K4
S_mix = c1*S1 + c2*S2 + c3*S3 + c4*S4
```
Where `c1..c4` are pigment concentrations (must sum to 1.0).

#### Step 2: Compute Reflectance (Equation 2)
```python
a = K / S
b = sqrt(a² + 2a)
R = 1 + a - b
```
This is the Kubelka-Munk reflectance equation for infinite thickness (substrate fully hidden).

#### Step 3: Saunderson Correction (Equation 6)
```python
R' = (1 - k1)(1 - k2) * R / (1 - k2 * R)
```
Accounts for surface reflection (Fresnel effects on the plastic surface).

#### Step 4: Reflectance → XYZ (Equations 3-5)
```python
X = ∫ R(λ) * D65(λ) * x̄(λ) dλ
Y = ∫ R(λ) * D65(λ) * ȳ(λ) dλ
Z = ∫ R(λ) * D65(λ) * z̄(λ) dλ
```
Integrates reflectance against CIE 1931 observer functions under D65 illuminant.

#### Step 5: XYZ → sRGB (Equation 7)
```python
[R, G, B] = M_xyz_to_srgb @ [X, Y, Z]
```
Transform matrix plus gamma correction.

### 3. Mixbox (The Gold Standard)

What makes Mixbox better than this implementation:

| Feature | This Project | Mixbox |
|---------|-------------|---------|
| Pigment Data | Gaussian approximations (Automated) | Measured from real paint samples |
| Optimization | Differentiable (scipy.optimize) | Neural network trained encoder/decoder |
| Runtime Performance | SLSQP solver (~5ms) | LUT lookup (~0.1ms) |
| Gamut Coverage | RGB residual for out-of-gamut | Perceptually optimized across full gamut |
| License | MIT (commercial friendly) | CC BY-NC 4.0 (non-commercial only) |

**Why Mixbox is better:**
1. **Real spectral data** — K/S curves from physical paint measurements, not mathematical guesses
2. **End-to-end training** — LUT is jointly optimized to minimize perceptual error
3. **LUT precomputation** — All mixing is instant (just array lookups)

## Pigment Design and Parameters

Each filament is defined in `pigments.py` using Gaussian peaks:

```python
K = baseline + gaussian_peak(wavelengths, center, width, height)
S = constant_scattering
```

### Tunable Parameters

| Parameter | Effect | Example |
|-----------|--------|---------|
| **center** | Dominant wavelength (hue) | Cyan: 640nm (absorbs red) |
| **width** | Spectral purity | Narrow = laser-focused, Wide = broad pigment |
| **height** | Absorption strength | High = dominates mixes, Low = subtle tint |
| **S baseline** | Opacity/milkiness | White: high S (scatters all light) |
| **k1, k2** | Surface reflection | Saunderson coefficients (default: 0.04, 0.6) |

### Critical Tuning Examples from Git History

**Cyan filament** (lines 25-31 in `pigments.py`):
- **Problem:** Original wide peaks "bled" absorption into green (530nm), darkening Blue+Yellow mixes
- **Fix:** Narrowed red absorption (640nm, width=60) and orange absorption (600nm, width=30)
- **Result:** Green channel stays clean, Blue+Yellow → vibrant green

**White filament** (lines 79-81):
- **Problem:** S=2.5 was "overwhelming" other pigments, all X+White mixes turned flat pastel
- **Fix:** Lowered S from 2.5 → 1.2
- **Result:** White adds lightness without crushing saturation

**Magenta filament** (lines 42-47):
- **Dual-peak strategy:** Narrow peak at 530nm (don't leak into blue) + wider peak at 555nm (extend absorption toward yellow-green)
- **Why asymmetric:** Preserves blue purity in Red+White while fully eating green for Red+Cyan → Purple

**Automated Optimization (White Pigment)**:
- **Discovery:** The optimizer (`scripts/optimize_pigments.py`) found that a much darker/clearer White (`K=0.099`, `S=0.5`) matched Mixbox better than the manual guess (`K=0.008`, `S=1.2`).
- **Impact:** Massive improvement in Red+White mixing accuracy (dE 14.2 → 7.8) without degrading other tints.

## API Usage

### Production Models (Recommended)

```python
from filament_mixer import PolyMixer, GPMixer

# Option 1: PolyMixer - Fastest (0.001ms, dE 2.07)
poly = PolyMixer.from_cache("models")
green = poly.lerp(0, 33, 133,  252, 211, 0,  0.5)  # Blue + Yellow
print(f"RGB{green}")  # Vibrant green!

# Option 2: GPMixer - Most Accurate (0.018ms, dE 1.79)
gp = GPMixer.from_cache("models")
green = gp.lerp(0, 33, 133,  252, 211, 0,  0.5)
print(f"RGB{green}")  # Even more accurate!
```

### Physics Engine (Research/N-way Mixing)

```python
from filament_mixer import FilamentMixer, CMYW_PALETTE

mixer = FilamentMixer(CMYW_PALETTE)

# Mix two colors
green = mixer.lerp(0, 33, 133,  252, 211, 0,  0.5)  # Blue + Yellow
print(f"RGB{green}")  # Vibrant green, not gray!

# Mix N colors with arbitrary weights
purple = mixer.mix_n_colors(
    colors=[(0, 33, 133), (252, 211, 0), (255, 255, 255)],
    weights=[0.5, 0.3, 0.2]
)
```

### Inverse Problem (Get Filament Ratios)

```python
# What percentages of CMYW give me this orange?
ratios = mixer.get_filament_ratios(255, 128, 0)
# ratios = [C%, M%, Y%, W%]

# Generate G-code for 4-in-1-out hotend
for i, ratio in enumerate(ratios):
    print(f"M163 S{i} P{ratio:.6f}")
print("M164 S0")
```

### Latent Space Interpolation

Colors are encoded as 7D vectors: `[c1, c2, c3, c4, rR, rG, rB]`
- First 4 components: pigment concentrations
- Last 3 components: RGB residual (for out-of-gamut correction)

```python
latent1 = mixer.rgb_to_latent(255, 0, 0)    # Red
latent2 = mixer.rgb_to_latent(0, 0, 255)    # Blue
mid = (latent1 + latent2) / 2
purple = mixer.latent_to_rgb(mid)           # Natural purple gradient
```

## Built-in Palettes

```python
CMYW_PALETTE  # Cyan, Magenta, Yellow, White — widest gamut
CMYK_PALETTE  # Cyan, Magenta, Yellow, Black — deep darks
RYBW_PALETTE  # Red, Yellow, Blue, White — traditional art mixing
```

**Why CMYW is recommended:** Wider color gamut than RYB, plus White allows tinting without desaturation.

## Known Limitations (Physics Engine)

1. **Approximate spectra** — Gaussian peaks, not measured filament data
2. **Runtime overhead** — `unmix()` solves constrained optimization on every call (~8ms)
3. **Lower accuracy** — Physics model achieves dE ~11.77 vs Mixbox
4. **Out-of-gamut handling** — Uses linear RGB residual (works but not optimal)

**Production Solutions:**
- **PolyMixer** — 4th-degree polynomial trained on Mixbox (dE 2.07, 0.001ms)
- **GPMixer** — Gaussian Process trained on Mixbox (dE 1.79, 0.018ms)
- Both models vastly outperform the physics engine while maintaining physically plausible behavior

## Comparison to Alternatives

| Approach | Speed | Accuracy (dE vs Mixbox) | Use Case |
|----------|-------|-------------------------|----------|
| Naive RGB | Instant | ~35.0 | Legacy slicers (incorrect physics) |
| GPMixer (Prod) | **0.018ms** | **1.79** | Best accuracy, production ready |
| PolyMixer (Prod) | **0.001ms** | **2.07** | Fastest, production ready |
| Wondermixer (Exp E) | 0.10ms | 3.38 | Numpy-only, embedded systems |
| Physics (K-M) | ~8ms | 11.77 | Research / N-way mixing |
| Physics (256³ LUT) | 0.02ms | 11.77 | Accelerated physics |
| Mixbox (Reference) | 0.01ms | 0.00 | Gold standard |

## Future Improvements (for Physics Engine)

Path to improve the physics-based approach:

1. **Measure real K/S curves** — Use spectrophotometer on physical filament samples
2. **Train encoder/decoder** — Optimize pigment concentrations against measured data
3. **Perceptual loss** — Minimize ΔE (color difference) across full gamut

Note: The learned models (PolyMixer, GPMixer) already achieve near-Mixbox accuracy without these improvements.

## Project Status

**Production Ready.** Two production models available:
- **PolyMixer** — 4th-degree polynomial (dE 2.07, 0.001ms) — Fastest option
- **GPMixer** — Gaussian Process (dE 1.79, 0.018ms) — Most accurate option

Both models produce physically plausible results (blue + yellow → green, not gray) with accuracy approaching Mixbox (dE 1.79 vs Mixbox's 0.00). Suitable for production use in 3D printer slicers and color mixing applications.

The physics engine (Kubelka-Munk) remains available for research, N-way mixing, and spectral analysis.

**License:** MIT (safe for commercial use, unlike Mixbox's CC BY-NC 4.0)
