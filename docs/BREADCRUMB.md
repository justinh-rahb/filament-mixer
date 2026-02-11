# Project Breadcrumb: FilamentMixer

## Current State (2026-02-11)
- **Production Performance:** Two production models available:
  - **PolyMixer (Experiment A)**: 4th-degree polynomial, dE **2.07** at **0.001ms** per mix
  - **GPMixer (Experiment C)**: Gaussian Process, dE **1.79** (best accuracy) at **0.018ms** per mix
- **Recent Research:** Experiment E (Wondermixer) evaluated - numpy-only cubic polynomial with dE 3.38
- **Physics Core:** Kubelka-Munk implementation remains available for N-way mixing and research (dE 11.77)
- **Benchmarks:** Comprehensive suite in `benchmarks/` directory

## 🚫 Failed Experiments (Do Not Repeat)
- **Narrowing Yellow Peak (< 45nm width):** Attempted to reduce green overlap by narrowing Yellow's blue absorption. Resulted in **Blue leakage** into the green spectrum, worsening the mix (dE spiked to ~35).
- **Aggressive Spectral Separation:** Attempted to shift Cyan (> 600nm) and Yellow (< 440nm) apart to avoid any overlap. Resulted in a **spectral "hole"** (500-550nm passband too wide), causing muddy/greyish mixes instead of vibrant greens.
- **Conclusion:** The optimal balance is **Cyan** with narrow peaks (Red=640/45, Orange=600/20) and **Yellow** with standard bandwidth (Blue=450/45).

## Knowledge Base
- **`docs/CLAUDE.md`:** Full technical architecture, math details, and API guide
- **`docs/EXPERIMENTS.md`:** Research experiments A-E with results and conclusions
- **`docs/LUT_GENERATION.md`:** LUT generation process and usage

## Workspace Info
- **Environment:** `venv` is set up with all dependencies (numpy, scipy, matplotlib, pillow, pymixbox).
- **Command:** `export PYTHONPATH=$PYTHONPATH:$(pwd)/src && source venv/bin/activate`
