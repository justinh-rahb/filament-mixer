# Project Breadcrumb: FilamentMixer

## Current State (2026-02-11)
- **Production Performance:** The **PolyMixer (Experiment A)** has been productionized. It uses a 4th-degree polynomial regression to achieve a Mean Delta-E of **2.07** (vs 11.77 for pure physics) at **~0.001ms** per mix.
- **Physics Core:** Implementation of Kubelka-Munk theory remains solid for N-way mixing and research. A 256³ high-resolution LUT (dE 11.77) is available for physics-based acceleration.
- **Benchmarks:** Comprehensive suite including `compare_poly.py` and visual comparisons.

## 🚫 Failed Experiments (Do Not Repeat)
- **Narrowing Yellow Peak (< 45nm width):** Attempted to reduce green overlap by narrowing Yellow's blue absorption. Resulted in **Blue leakage** into the green spectrum, worsening the mix (dE spiked to ~35).
- **Aggressive Spectral Separation:** Attempted to shift Cyan (> 600nm) and Yellow (< 440nm) apart to avoid any overlap. Resulted in a **spectral "hole"** (500-550nm passband too wide), causing muddy/greyish mixes instead of vibrant greens.
- **Conclusion:** The optimal balance is **Cyan** with narrow peaks (Red=640/45, Orange=600/20) and **Yellow** with standard bandwidth (Blue=450/45).

## Knowledge Base
- **[`CLAUDE.md`](file:///Users/justinh/Development/github.com/justinh-rahb/filament-mixer/CLAUDE.md):** Full technical architecture, math details, and API guide.
- **[`walkthrough.md`](file:///Users/justinh/.gemini/antigravity/brain/7f05c618-69be-4a4b-81d8-a005d536b754/walkthrough.md):** Visual proof, recent benchmark analysis, and historical tuning context.

## Workspace Info
- **Environment:** `venv` is set up with all dependencies (numpy, scipy, matplotlib, pillow, pymixbox).
- **Command:** `export PYTHONPATH=$PYTHONPATH:$(pwd)/src && source venv/bin/activate`
