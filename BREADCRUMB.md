# Project Breadcrumb: FilamentMixer

## Current State (2026-02-10)
- **Physics Core:** Implementation of Kubelka-Munk theory (Equations 1-7) is solid and verified.
- **Performance:** A 256³ high-resolution LUT has been generated and integrated. Speed is now **~0.02ms** per mixing operation (347x speedup).
- **Benchmarks:** Official `pymixbox` is integrated.
- **Accuracy:** Mean Delta-E vs Mixbox is **12.87** (down from 14.4). Blue+Yellow saturation is significantly improved (dE 29.79).

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
