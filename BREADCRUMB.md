# Project Breadcrumb: FilamentMixer

## Current State (2026-02-10)
- **Physics Core:** Implementation of Kubelka-Munk theory (Equations 1-7) is solid and verified.
- **Performance:** A 256³ high-resolution LUT has been generated and integrated. Speed is now **~0.02ms** per mixing operation (400x speedup vs math-on-the-fly).
- **Benchmarks:** Official `pymixbox` is installed and integrated into the bench suite.
- **Accuracy baseline:** Mean Delta-E vs Mixbox is **14.4**. We match the "hue" (Blue+Yellow=Green) but lag behind on "saturation/vibrancy".

## Knowledge Base
- **[`CLAUDE.md`](file:///Users/justinh/Development/github.com/justinh-rahb/filament-mixer/CLAUDE.md):** Full technical architecture, math details, and API guide.
- **[`walkthrough.md`](file:///Users/justinh/.gemini/antigravity/brain/7f05c618-69be-4a4b-81d8-a005d536b754/walkthrough.md):** Visual proof, recent benchmark analysis, and historical tuning context.

## 🚩 Priority Next Steps
1. **Saturation Tuning:** Narrow the Gaussian peaks for Cyan and Magenta in `src/filament_mixer/pigments.py`. Mixbox's green is significantly more vibrant; our model is currently too "safe/absorbing".
2. **G-Code Integration:** Develop the script to apply spectral mixing ratios to real-world `M163` (Prusa/Marlin multi-extruder) commands.
3. **Material System:** Implement `materials.json` to allow switching between filament brands with different spectral properties.

## Workspace Info
- **Environment:** `venv` is set up with all dependencies (numpy, scipy, matplotlib, pillow, pymixbox).
- **Command:** `export PYTHONPATH=$PYTHONPATH:$(pwd)/src && source venv/bin/activate`
