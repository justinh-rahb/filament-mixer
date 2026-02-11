# FilamentMixer: Physics-Based Color Mixing for 3D Printing

## Project Purpose

FilamentMixer implements physically-accurate color mixing for multi-material 3D printers. It replaces naive RGB interpolation (which produces muddy, desaturated blends) with spectral-based pigment physics using Kubelka-Munk theory.

**The core problem:** Traditional slicers use `(Color A + Color B) / 2` to calculate mixing ratios. This treats colors like light beams (additive), but physical pigments mix **subtractively** — they absorb and scatter light. Blue + Yellow should give vibrant Green, not gray.

**The solution:** Model each filament as a physical pigment with wavelength-dependent absorption (K) and scattering (S) properties, mix in spectral space, then convert back to RGB.

## Architecture Overview

```
src/filament_mixer/
├── km_core.py       # Kubelka-Munk physics engine (K-M equations 1-7)
├── pigments.py      # Spectral definitions for CMYW/CMYK/RYBW palettes
├── unmixer.py       # Inverse solver (RGB → pigment concentrations)
├── api.py           # FilamentMixer class (main entry point)
└── lut.py           # Look-up table generator for fast runtime mixing
```

### Key Classes

- **`Pigment`** — Dataclass containing name, K spectrum (38 wavelengths), S spectrum
- **`KubelkaMunk`** — Physics engine that implements the full mixing pipeline
- **`RGBUnmixer`** — Constrained optimizer (SLSQP) that solves RGB → concentrations
- **`FilamentMixer`** — High-level API that combines all components

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
| Pigment Data | Gaussian approximations | Measured from real paint samples |
| Optimization | None (hand-tuned) | Neural network trained encoder/decoder |
| Runtime Performance | SLSQP solver (~10-50ms) | LUT lookup (~0.1ms) |
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

## API Usage

### Basic Mixing

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

## Known Limitations

1. **Approximate spectra** — Gaussian peaks, not measured filament data
2. **Runtime overhead** — `unmix()` solves constrained optimization on every call (slow)
3. **No perceptual training** — Hand-tuned, not optimized against human perception
4. **Out-of-gamut handling** — Uses linear RGB residual (works but not optimal)
5. **The "grey problem"** — Plain K-M without careful spectral tuning gets muddy mixes

**Mitigations:**
- Gaussian peaks are better than binary step functions (why we avoid the worst "grey problem")
- Carefully tuned spectra based on real pigment behavior (Phthalo Blue, Quinacridone Magenta, etc.)
- LUT precomputation available via `lut.py` (trades memory for speed)

## Comparison to Alternatives

| Approach | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| Naive RGB | Instant | Poor | Legacy slicers |
| This (K-M + Gaussian) | ~10-50ms | Good | 3D printing (MIT licensed) |
| Mixbox (K-M + Measured + LUT) | <1ms | Excellent | Digital painting (non-commercial) |

## Future Improvements

From `README.md` limitations section, the path to Mixbox-level quality:

1. **Measure real K/S curves** — Use spectrophotometer on physical filament samples
2. **Train encoder/decoder** — Optimize pigment concentrations against measured data
3. **Precompute LUT** — Generate 256³ table from trained model
4. **Perceptual loss** — Minimize ΔE (color difference) across full gamut

## Project Status

**Alpha / Research.** Produces noticeably better results than naive RGB (blue + yellow → green, not gray), but not tuned to Mixbox's level. Good enough for practical 3D printer color mixing; needs work for professional color reproduction.

**License:** MIT (safe for commercial use, unlike Mixbox's CC BY-NC 4.0)
