#!/usr/bin/env python3
"""
Slicer Integration Example

Demonstrates how to use FilamentMixer in a real slicer workflow,
generating M163/M164 G-code commands for multi-material full-color printing.
"""

import numpy as np

from filament_mixer import FilamentMixer, CMYW_PALETTE, rgb_to_uint8


class FullColorGCodeGenerator:
    """
    Example G-code generator for full-color printing.

    Simulates what the Orca Slicer fork would do with FilamentMixer.
    """

    def __init__(self, palette="CMYW"):
        if palette == "CMYW":
            self.palette = CMYW_PALETTE
            self.filament_names = ["Cyan", "Magenta", "Yellow", "White"]

        self.mixer = FilamentMixer(self.palette)
        self.current_mix = np.array([0.25, 0.25, 0.25, 0.25])

    def set_color(self, r: int, g: int, b: int) -> str:
        """Generate G-code to set the extruder mix ratio for a target color."""
        ratios = self.mixer.get_filament_ratios(r, g, b)

        gcode = []
        gcode.append(f"; Target color: RGB({r}, {g}, {b})")

        for i, (name, ratio) in enumerate(zip(self.filament_names, ratios)):
            gcode.append(f"M163 S{i} P{ratio:.6f}  ; {name}: {ratio * 100:.2f}%")

        gcode.append("M164 S0                  ; Save to virtual extruder 0")

        self.current_mix = ratios
        return "\n".join(gcode)

    def gradient_layer(self, start_rgb, end_rgb, num_layers=10):
        """Generate G-code for a gradient across multiple layers."""
        print(f"\nGenerating {num_layers}-layer gradient")
        print(f"  Start: RGB{start_rgb}")
        print(f"  End:   RGB{end_rgb}")
        print("=" * 60)

        for layer in range(num_layers):
            t = layer / (num_layers - 1)
            current_rgb = self.mixer.lerp(*start_rgb, *end_rgb, t)

            print(f"\n; Layer {layer} (t={t:.2f})")
            print(self.set_color(*current_rgb))
            print(f"G1 Z{layer * 0.2:.2f}  ; Move to layer height")
            print("; ... extrusion moves here ...")


def demo_basic_colors():
    gen = FullColorGCodeGenerator()

    print("\n" + "=" * 60)
    print("DEMO 1: Basic Color G-code Generation")
    print("=" * 60)

    test_colors = [
        ("Red", (255, 0, 0)),
        ("Green", (0, 255, 0)),
        ("Blue", (0, 0, 255)),
        ("Orange", (255, 128, 0)),
        ("Purple", (128, 0, 128)),
        ("Skin tone", (255, 220, 177)),
    ]

    for name, rgb in test_colors:
        print(f"\n{name}:")
        print(gen.set_color(*rgb))


def demo_gradient():
    gen = FullColorGCodeGenerator()

    print("\n" + "=" * 60)
    print("DEMO 2: Gradient Across Layers")
    print("=" * 60)

    gen.gradient_layer((0, 33, 133), (252, 211, 0), num_layers=10)


def demo_comparison():
    mixer = FilamentMixer(CMYW_PALETTE)

    print("\n" + "=" * 60)
    print("DEMO 3: Pigment Mixing vs RGB Comparison")
    print("=" * 60)

    blue = (0, 33, 133)
    yellow = (252, 211, 0)

    print(f"\nMixing Blue RGB{blue} + Yellow RGB{yellow} at t=0.5")
    print("-" * 60)

    rgb_result = tuple(int((1 - 0.5) * blue[i] + 0.5 * yellow[i]) for i in range(3))
    mixbox_result = mixer.lerp(*blue, *yellow, 0.5)

    print(f"\nRGB Lerp:        RGB{rgb_result}")
    print(f"  -> Muddy gray-brown")
    print(f"\nFilamentMixer:   RGB{mixbox_result}")
    print(f"  -> Vibrant green!")

    print(f"\nFilament ratios for the mixed green:")
    ratios = mixer.get_filament_ratios(*mixbox_result)
    for name, r in zip(["Cyan", "Magenta", "Yellow", "White"], ratios):
        print(f"  {name:10s}: {r * 100:5.1f}%")


def demo_dithering_preview():
    mixer = FilamentMixer(CMYW_PALETTE)

    print("\n" + "=" * 60)
    print("DEMO 4: Layer Dithering for Smooth Gradients")
    print("=" * 60)

    target = (128, 200, 80)
    ratios = mixer.get_filament_ratios(*target)

    print(f"\nTarget color: RGB{target}")
    print(
        f"Exact ratios: C={ratios[0]:.3f}, M={ratios[1]:.3f}, "
        f"Y={ratios[2]:.3f}, W={ratios[3]:.3f}"
    )

    print(f"\nFor 0.1mm layers, using variable layer heights:")
    for name, r in zip(["Cyan", "Magenta", "Yellow", "White"], ratios):
        print(f"  {name} layer: {r * 0.1:.4f}mm")
    print(f"  Total:         {sum(ratios) * 0.1:.4f}mm")

    print("\nWith fixed 0.08mm layers, dithering over 10 layers:")
    layers_per_color = (ratios * 10).astype(int)
    for name, n in zip(["Cyan", "Magenta", "Yellow", "White"], layers_per_color):
        print(f"  {n} layers {name}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" FILAMENT MIXER - SLICER INTEGRATION DEMO")
    print("=" * 60)

    demo_basic_colors()
    demo_gradient()
    demo_comparison()
    demo_dithering_preview()

    print("\n" + "=" * 60)
    print("Done! Integration demo complete.")
    print("=" * 60)
