# FilamentMixer C++

Header-only C++ port of the PolyMixer (Experiment A) — a degree-4 polynomial regression trained to approximate [Mixbox](https://github.com/scrtwpns/mixbox) behavior. Drop-in replacement for `mixbox_lerp`.

This library is an independent approximation and does not include Mixbox source code, binaries, or data files.

**Performance:** Mean Delta-E **2.07** vs Mixbox reference. Blue + Yellow gives you **green**, not gray.

## Why C++?

The Python `PolyMixer` is already the fastest mixer in the project (0.001ms per mix), but it still requires Python + sklearn at runtime to load the pickle model. This C++ port bakes the trained coefficients directly into a header file — no runtime dependencies, no model loading, no Python. Just `#include` and go.

Useful for:
- Embedding in C/C++ slicer plugins (PrusaSlicer, Cura, OrcaSlicer)
- Firmware-level mixing on 3D printer control boards
- Any environment where Python isn't available or desirable

## Quick Start

```cpp
#include "filament_mixer.h"

// Pointer-out style (matches mixbox_lerp signature)
unsigned char r, g, b;
filament_mixer::lerp(0, 33, 133,  252, 211, 0,  0.5f, &r, &g, &b);
// r=47, g=141, b=56  (blue + yellow -> green)

// Or use the struct-return convenience overload
auto rgb = filament_mixer::lerp(0, 33, 133,  252, 211, 0,  0.5f);
printf("(%d, %d, %d)\n", rgb.r, rgb.g, rgb.b);
```

### Replacing Mixbox

If you're currently using mixbox's C API, the migration is mechanical:

```cpp
// Before (mixbox)
#include "mixbox.h"
mixbox_lerp(r1, g1, b1, r2, g2, b2, t, &r, &g, &b);

// After (filament_mixer)
#include "filament_mixer.h"
filament_mixer::lerp(r1, g1, b1, r2, g2, b2, t, &r, &g, &b);
```

Same parameter order, same types, same semantics. The only difference is the function name and namespace.

## Build

### Requirements

- C++11 or later
- No external dependencies (just `<cmath>`, `<cstdint>`, `<algorithm>`)

### With CMake

```bash
cd cpp
cmake -B build .
cmake --build build
./build/filament_mixer_example
```

### Without CMake

It's a single header — just compile directly:

```bash
cd cpp
c++ -std=c++11 -O2 -o example example.cpp
./example
```

### In Your Own Project

Copy this file into your project:
- `filament_mixer.h` — the library with trained coefficients inlined

Then `#include "filament_mixer.h"` wherever you need color mixing.

## API Reference

### `filament_mixer::lerp` (pointer-out)

```cpp
void filament_mixer::lerp(
    unsigned char r1, unsigned char g1, unsigned char b1,  // First color (0-255)
    unsigned char r2, unsigned char g2, unsigned char b2,  // Second color (0-255)
    float t,                                                // Mixing ratio [0, 1]
    unsigned char* out_r, unsigned char* out_g, unsigned char* out_b  // Output
);
```

Mixes two RGB colors using polynomial pigment mixing. This is the drop-in replacement for `mixbox_lerp`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `r1, g1, b1` | `unsigned char` | First color, each channel 0-255 |
| `r2, g2, b2` | `unsigned char` | Second color, each channel 0-255 |
| `t` | `float` | Mixing ratio. 0.0 = 100% color1, 1.0 = 100% color2 |
| `out_r, out_g, out_b` | `unsigned char*` | Output color channels (written to) |

**Endpoint behavior:** When `t <= 0`, returns color1 exactly. When `t >= 1`, returns color2 exactly. No polynomial evaluation is performed at the endpoints — the input colors are returned as-is.

### `filament_mixer::lerp` (struct-return)

```cpp
filament_mixer::RGB filament_mixer::lerp(
    unsigned char r1, unsigned char g1, unsigned char b1,
    unsigned char r2, unsigned char g2, unsigned char b2,
    float t
);
```

Same as above, but returns the result as an `RGB` struct instead of writing to pointers.

### `filament_mixer::RGB`

```cpp
struct RGB {
    unsigned char r, g, b;
};
```

Simple POD struct for returning color values.

## How It Works

The model is a degree-4 polynomial regression mapping 7 inputs to 3 outputs:

```
Input:  [R1, G1, B1, R2, G2, B2, t]  (7 values)
                    |
        Polynomial Feature Expansion
        (all monomials up to degree 4)
                    |
        330 polynomial features
                    |
        Linear regression (330x3 matrix multiply + bias)
                    |
Output: [R_mix, G_mix, B_mix]  (3 values, clamped to 0-255)
```

### Polynomial Features

For 7 input variables at degree 4, `PolynomialFeatures` generates 330 terms:

| Degree | Terms | Examples |
|--------|-------|---------|
| 0 | 1 | `1` (bias) |
| 1 | 7 | `r1`, `g1`, `b1`, `r2`, `g2`, `b2`, `t` |
| 2 | 28 | `r1*r1`, `r1*g1`, `r1*b1`, ..., `t*t` |
| 3 | 84 | `r1*r1*r1`, `r1*r1*g1`, ..., `t*t*t` |
| 4 | 210 | `r1^4`, `r1^3*g1`, ..., `t^4` |
| **Total** | **330** | |

Each feature is a monomial `x0^a0 * x1^a1 * ... * x6^a6` where `a0+a1+...+a6 <= 4`. The exponent patterns are stored in the `POWERS[330][7]` table in `filament_mixer.h`.

### Coefficients

The trained coefficients are stored as:
- `COEF[330][3]` — weight matrix (330 features x 3 output channels)
- `INTERCEPT[3]` — bias vector (one per output channel)

The prediction is: `output[c] = INTERCEPT[c] + sum(features[i] * COEF[i][c])` for each channel `c` in {R, G, B}.

### Training

The coefficients were trained on 200,000 random color pairs mixed through [Mixbox](https://scrtwpns.com/mixbox/) (Sochorova & Jamriska, 2021) using scikit-learn's `PolynomialFeatures(degree=4) + LinearRegression`. The training process is documented in `scripts/train_poly_model.py`.

## Accuracy

Evaluated against Mixbox ground truth on 10,000 random test samples:

| Metric | Value |
|--------|-------|
| Mean Delta-E | **2.07** |
| Median Delta-E | ~1.6 |
| < 2.0 (imperceptible) | ~60% of samples |
| < 5.0 (minor difference) | ~95% of samples |

### Signature Test: Blue + Yellow

The classic pigment mixing test — this should produce green, not the muddy gray that naive RGB interpolation gives.

| Ratio | FilamentMixer C++ | Mixbox Reference | Delta-E |
|-------|-------------------|------------------|---------|
| t=0.25 | `(28, 80, 171)` | `(25, 72, 173)` | ~3.5 |
| t=0.50 | `(47, 141, 56)` | `(41, 130, 57)` | ~4.7 |
| t=0.75 | `(161, 200, 18)` | `(153, 195, 15)` | ~3.4 |

All results are perceptually green — the polynomial successfully captures the subtractive mixing behavior.

## Regenerating Coefficients

If you retrain the Python model (different degree, more samples, etc.), regenerate the inlined C++ coefficients:

```bash
# From the project root
python scripts/export_poly_coefficients.py
```

This reads `models/poly_model.pkl`, extracts the polynomial powers and regression coefficients, verifies them against sklearn's predictions, and updates the auto-generated coefficient block in `cpp/filament_mixer.h`.

## File Structure

```
cpp/
├── filament_mixer.h          # Header-only library with auto-generated coefficients
├── example.cpp               # Test program with verification cases
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## Comparison to Mixbox

| | FilamentMixer C++ | Mixbox C++ |
|---|---|---|
| **Accuracy** (dE vs Mixbox) | 2.07 | 0.00 (reference) |
| **Method** | Polynomial regression | LUT + latent space |
| **Header size** | ~36 KB (coefficients) | ~500 KB (LUT data) |
| **Dependencies** | None | None |
| **License** | MIT | CC BY-NC 4.0 |
| **C++ standard** | C++11 | C++11 |

The accuracy difference (dE 2.07) is below the threshold of casual perception for most color pairs. The main advantage is the MIT license — no commercial licensing restrictions.

## License

MIT — same as the parent FilamentMixer project.

The polynomial coefficients were trained against Mixbox outputs as reference targets. This repository does not include Mixbox source code, binaries, or data files. Mixbox itself is by Sarka Sochorova and Ondrej Jamriska ([scrtwpns.com/mixbox](https://scrtwpns.com/mixbox/)).
