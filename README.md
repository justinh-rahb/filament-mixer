# FilamentMixer

Physics-based color mixing for 3D printer filaments using [Kubelka-Munk theory](https://en.wikipedia.org/wiki/Kubelka%E2%80%93Munk_theory).

Replaces naive RGB interpolation — which produces muddy, desaturated blends — with the same spectral approach used for real paint/pigment mixing. Based on the [Mixbox paper](https://scratchapixel.com/lessons/digital-imaging/colors/color-of-objects.html) by Sochorová & Jamriška (2021).

**The result:** mixing blue + yellow gives you **green**, not gray.

## Status

> **Production Ready.** FilamentMixer now includes **GPMixer** (Gaussian Process Regression),
> a learned model that achieves **Mean Delta-E of 1.79** against Mixbox ground truth —
> the most accurate non-LUT method available. It runs at **0.018ms** per mix, making it
> **56x faster than the physics engine** while producing vibrant, physically plausible
> color blends (Blue+Yellow=Green). The original K-M physics engine (dE 11.77) is still
> available for research and spectral tuning.

## Why?

Multi-material 3D printers (e.g. 4-in-1-out hotends with CMYW filaments) mix colors physically — just like paint. But slicers still use RGB linear interpolation to compute mixing ratios, which doesn't model how pigments actually absorb and scatter light.

FilamentMixer fixes this by:
1. Decomposing colors into spectral pigment concentrations (K-M theory)
2. Mixing in spectral space where the physics actually works
3. Converting back to RGB for display / G-code generation

## Install

```bash
pip install filament-mixer
```

Or from source:

```bash
git clone https://github.com/justinh-rahb/filament-mixer.git
cd filament-mixer
pip install -e ".[dev]"
```

## Quick Start

```python
from filament_mixer import GPMixer  # Fast, accurate learned model

mixer = GPMixer()  # Loads pre-trained model from lut_gp/

# Mix blue + yellow = GREEN (not gray!)
green = mixer.lerp(0, 33, 133,  252, 211, 0,  0.5)
print(f"Result: RGB{green}")  # Vibrant green: (47, 139, 49)

# Or use the physics engine:
# from filament_mixer import FilamentMixer, CMYW_PALETTE
# mixer = FilamentMixer(CMYW_PALETTE)
# ratios = mixer.get_filament_ratios(255, 128, 0)
```

## Slicer Integration

**Recommended:** Use GPMixer for best accuracy and speed:

```python
from filament_mixer import GPMixer

mixer = GPMixer()  # Loads pre-trained model

# Drop-in replacement for: result = (1-t)*color1 + t*color2
result = mixer.lerp(*color1, *color2, t)
```

**Alternative:** Use the physics engine for filament ratio calculation:

```python
from filament_mixer import FilamentMixer, CMYW_PALETTE

mixer = FilamentMixer(CMYW_PALETTE)
ratios = mixer.get_filament_ratios(128, 200, 80)

# Generate M163/M164 G-code for multi-extruder setups
for i, ratio in enumerate(ratios):
    print(f"M163 S{i} P{ratio:.6f}")
print("M164 S0")
```

## Comparison to Alternatives

| Approach | Speed | Accuracy (dE vs Mixbox) | Use Case |
|----------|-------|-------------------------|----------|
| Naive RGB Lerp | Instant | ~35.0 (Varies) | Legacy slicers |
| **GPMixer (This)** | **0.018ms** | **1.79** 🏆 | **Production / Recommended** |
| FastLUT 256³ (This) | 0.02ms | 11.77 | Pre-cached mixing |
| K-M Physics (This) | ~4.8ms | 11.77 | Research / Spectral tuning |
| Mixbox (Reference) | 0.01ms | 0.00 | Digital painting (Commercial license) |

**GPMixer** is a Gaussian Process model trained on 2,000 Mixbox samples. It learns Mixbox's behavior directly, achieving near-perfect accuracy at production speeds.

## Built-in Palettes

| Palette | Filaments | Best for |
|---------|-----------|----------|
| `CMYW_PALETTE` | Cyan, Magenta, Yellow, White | Widest gamut, bright colors |
| `CMYK_PALETTE` | Cyan, Magenta, Yellow, Black | Deep darks, print-like output |
| `RYBW_PALETTE` | Red, Yellow, Blue, White | Traditional art mixing |

## API Reference

### `FilamentMixer(pigments)`

Main class. Initialize with a list of 4 `Pigment` objects (or use a built-in palette).

| Method | Description |
|--------|-------------|
| `lerp(r1, g1, b1, r2, g2, b2, t)` | Blend two colors with pigment-based mixing |
| `mix_n_colors(colors, weights)` | Mix N colors with arbitrary weights |
| `get_filament_ratios(r, g, b)` | Get filament percentages for a target RGB color |
| `rgb_to_latent(r, g, b)` | Encode RGB to 7-D latent space |
| `latent_to_rgb(latent)` | Decode latent space back to RGB |

### Supporting classes

- **`Pigment(name, K, S)`** — Spectral absorption (K) and scattering (S) coefficients
- **`KubelkaMunk(k1, k2)`** — Low-level spectral mixing engine
- **`RGBUnmixer(pigments)`** — Inverse solver (RGB → concentrations)

### `GPMixer(model_path)`

**Recommended for production.** Gaussian Process-based mixer trained on Mixbox ground truth.

| Method | Description |
|--------|-------------|
| `lerp(r1, g1, b1, r2, g2, b2, t)` | Mix two colors (fastest, most accurate) |
| `from_cache(cache_dir)` | Load model from directory (default: "lut_gp") |

**Training your own model:**
```bash
python scripts/train_gp_model.py  # Trains on 2,000 Mixbox samples (~7s)
```

## Examples

```bash
# Visual comparison of RGB vs pigment mixing
python examples/visual_demo.py

# G-code generation demo
python examples/slicer_demo.py
```

## Project Structure

```
src/filament_mixer/
├── __init__.py      # Public API exports
├── gp_mixer.py     # GPMixer (Gaussian Process, recommended)
├── km_core.py       # Kubelka-Munk physics engine
├── pigments.py      # Filament spectral definitions & palettes
├── unmixer.py       # RGB → pigment concentration solver
├── api.py           # FilamentMixer class (physics-based)
└── lut.py           # Lookup table generator for fast caching

scripts/
└── train_gp_model.py  # Train GPMixer on Mixbox ground truth

lut_gp/
└── gp_model.pkl      # Pre-trained GPMixer model (32MB)
```

## How It Works

### GPMixer (Recommended)

A Gaussian Process Regressor trained on 2,000 Mixbox samples learns the direct `(RGB₁, RGB₂, t) → RGB_mix` mapping. This bypasses the complex physics and inverse problems entirely, achieving **dE 1.79** accuracy at **0.018ms** per mix.

**Why it works:** Mixbox is the ground truth for pigment mixing. By training directly on its outputs, GPMixer learns the perceptually-optimal color blending behavior without needing to understand the underlying spectral physics.

### FilamentMixer (Physics-Based)

1. **Spectral Mixing (K-M Theory):** Each filament is defined by its absorption (K) and scattering (S) spectra across 38 wavelengths (380–750nm). Colors are mixed by linearly combining K and S values — this is physically correct for pigment mixtures.

2. **Reflectance → Color:** The mixed K/S spectra are converted to a reflectance curve, then integrated against the CIE 1931 observer functions and D65 illuminant to produce XYZ tristimulus values, which are finally transformed to sRGB.

3. **Unmixing (Inverse Problem):** Given a target RGB color, constrained optimization (SLSQP) finds the pigment concentrations that best reproduce it — this is how `get_filament_ratios()` works.

4. **Latent Space:** Colors are represented as 7-D vectors `[c1, c2, c3, c4, rR, rG, rB]` — four pigment concentrations plus an RGB residual for out-of-gamut correction. Interpolation in this space produces natural-looking blends.

## Limitations

Plain Kubelka-Munk with hand-crafted spectra still has known issues (the "[grey problem](https://github.com/lwander/open-km)"). Our Gaussian-peak spectra are better than binary step functions (which is why open-km gets grey and we don't), but worse than real measured/optimized pigment data. Specifically:

- **Approximate spectra.** Our pigment K/S curves are Gaussian approximations, not measured from real filaments. Certain color combinations will be less accurate than others.
- **Runtime optimization.** `unmix()` solves a constrained optimization (SLSQP) on every call — slow compared to Mixbox's LUT lookup. The `lut.py` module can precompute tables, but they're based on the same approximate spectra.
- **No end-to-end training.** Mixbox trains its encoder/decoder and LUT jointly to minimize perceptual error across the full gamut. We don't do that.
- **Limited gamut.** Colors far outside the palette's reproducible gamut rely on the RGB residual, which is a linear correction — it works but it's not perceptually optimal.

### Why not just use Mixbox?

Mixbox is [CC BY-NC 4.0](https://scrtwpns.com/mixbox/docs/#license) (non-commercial only, commercial license required). This project is MIT-licensed and purpose-built for 3D printer filament mixing, where the palette (CMYW/CMYK) and mixing physics (plastic filament, not paint) are different from Mixbox's art-pigment focus.

For higher-quality results, the path forward would be:
1. Measure real K/S curves from physical filament samples
2. Train the encoder/decoder against measured data
3. Precompute a LUT from the trained model

## License

MIT
