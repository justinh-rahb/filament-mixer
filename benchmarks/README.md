# Benchmarks

This directory contains benchmarking scripts to compare FilamentMixer against other color mixing methods.

## Available Benchmarks

### 1. `compare.py` - Basic Numerical Comparison
Standard benchmark comparing FilamentMixer vs RGB lerp vs Mixbox.

**Usage:**
```bash
python benchmarks/compare.py
```

**Output:**
- Console tables with color values, saturation, hue, Delta-E metrics
- Simple PNG image showing color swatches
- Speed measurements

---

### 2. `compare_with_lut.py` - LUT-Enhanced Comparison
Extended benchmark including FastLUTMixer performance.

**Prerequisites:**
```bash
# Generate LUT tables first
python scripts/generate_lut.py --resolution 64

# Or generate higher resolution for better accuracy
python scripts/generate_lut.py --resolution 256
```

**Usage:**
```bash
# Use 64³ LUT (default)
python benchmarks/compare_with_lut.py

# Use 256³ LUT for better accuracy
python benchmarks/compare_with_lut.py --lut-resolution 256
```

**Output:**
- All metrics from basic benchmark
- FastLUTMixer results and speed comparison
- Shows 432x speedup vs regular FilamentMixer

---

### 3. `visual_compare.py` - Visual Comparison Grids
Generates beautiful comparison images showing color mixing results side-by-side.

**Usage:**
```bash
# Generate both grid and gradient comparisons
python benchmarks/visual_compare.py

# Custom output paths
python benchmarks/visual_compare.py --grid my_grid.png --gradient my_gradients.png

# Larger swatches
python benchmarks/visual_compare.py --swatch-size 150

# Use 256³ LUT for better accuracy
python benchmarks/visual_compare.py --lut-resolution 256

# Skip gradient comparison
python benchmarks/visual_compare.py --skip-gradient
```

**Generates:**

1. **`visual_comparison.png`** - Swatch grid showing:
   - Input colors (Color 1, Color 2)
   - RGB Lerp result
   - FilamentMixer result
   - FastLUT result (if available)
   - Mixbox result (if available)
   
2. **`gradient_comparison.png`** - Smooth gradient strips showing:
   - 9-step gradients between color pairs
   - Side-by-side comparison of all methods
   - Perfect for seeing transition quality

---

## Test Color Pairs

All benchmarks use the same 7 test pairs from Mixbox's documentation:

1. **Blue + Yellow** → Should produce green (not muddy gray)
2. **Red + Blue** → Purple
3. **Red + Yellow** → Orange
4. **Magenta + Yellow** → Red-orange
5. **Blue + White** → Light blue tint
6. **Red + White** → Pink/light red
7. **Green + Magenta** → Dark neutral

---

## Current Results Summary

### Accuracy (Mean Delta-E vs Mixbox):
- **FilamentMixer**: 14.43 ✓ (best accuracy)
- **FastLUTMixer (64³)**: 17.89 (quantization artifacts)
- **RGB Lerp**: 35+ (worst)

### Speed:
- **RGB Lerp**: 0.00 ms
- **FilamentMixer**: 5.3 ms
- **FastLUTMixer**: 0.01 ms ⚡ **(432x faster than FilamentMixer)**
- **Mixbox**: 0.01 ms

### Wins vs Mixbox:
- **FilamentMixer**: 6/7 pairs closer than RGB
- **FastLUTMixer**: 6/7 pairs closer than RGB
- Both correctly produce green from blue+yellow ✓

---

## Generated Files

```
benchmarks/
├── compare.py              # Basic benchmark script
├── compare_with_lut.py     # LUT-enhanced benchmark
├── visual_compare.py       # Visual grid generator
├── comparison.png          # Old simple comparison
├── visual_comparison.png   # New swatch grid (770×1170px)
└── gradient_comparison.png # Gradient strips
```

---

## Tips

**For best visual results:**
1. Generate a 256³ LUT for higher accuracy (takes ~60 min):
   ```bash
   python scripts/generate_lut.py --resolution 256
   ```

2. Install Mixbox for complete comparison:
   ```bash
   pip install pymixbox
   ```

3. Use custom color pairs by editing `COLOR_PAIRS` in the scripts
