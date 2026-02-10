#!/usr/bin/env python3
"""
Visual Comparison: Pigment Mixing vs RGB Mixing

Creates side-by-side gradient strips and color wheels showing the
difference between traditional RGB interpolation and physically-based
pigment mixing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from filament_mixer import FilamentMixer, CMYW_PALETTE


def rgb_lerp(rgb1, rgb2, t):
    """Traditional RGB linear interpolation."""
    return tuple(int((1 - t) * rgb1[i] + t * rgb2[i]) for i in range(3))


def create_gradient(color1, color2, steps=50, mixer=None):
    """Create a color gradient (pigment-based if *mixer* is provided)."""
    gradient = []
    for i in range(steps):
        t = i / (steps - 1)
        if mixer is not None:
            rgb = mixer.lerp(*color1, *color2, t)
        else:
            rgb = rgb_lerp(color1, color2, t)
        gradient.append(np.array(rgb) / 255.0)
    return np.array(gradient).reshape(1, steps, 3)


def plot_comparison():
    """Create comparison plots and save to comparison.png."""
    mixer = FilamentMixer(CMYW_PALETTE)

    color_pairs = [
        ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
        ("Cyan + Magenta", (26, 102, 153), (152, 58, 88)),
        ("Magenta + Yellow", (152, 58, 88), (252, 211, 0)),
        ("Blue + White", (0, 33, 133), (244, 226, 194)),
    ]

    fig, axes = plt.subplots(len(color_pairs), 2, figsize=(14, 8))
    fig.suptitle("Pigment Mixing vs RGB Mixing", fontsize=16, fontweight="bold")

    steps = 100

    for idx, (name, color1, color2) in enumerate(color_pairs):
        # RGB gradient
        rgb_grad = create_gradient(color1, color2, steps)
        axes[idx, 0].imshow(rgb_grad, aspect="auto")
        axes[idx, 0].set_title(f"{name} - RGB Lerp (Muddy)")
        axes[idx, 0].axis("off")

        for ax_col, grad_mixer in [(0, None), (1, mixer)]:
            grad = create_gradient(color1, color2, steps, mixer=grad_mixer)
            axes[idx, ax_col].imshow(grad, aspect="auto")
            axes[idx, ax_col].axis("off")

            for pos, color in [(0, color1), (steps * 0.85, color2)]:
                rect = Rectangle(
                    (pos, 0.2),
                    steps * 0.15,
                    0.6,
                    facecolor=np.array(color) / 255.0,
                    edgecolor="white",
                    linewidth=2,
                )
                axes[idx, ax_col].add_patch(rect)

        axes[idx, 0].set_title(f"{name} - RGB Lerp (Muddy)")
        axes[idx, 1].set_title(f"{name} - Pigment Mixing (Natural)")

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=200, bbox_inches="tight")
    print("Saved comparison to comparison.png")

    create_color_wheel_comparison(mixer)


def create_color_wheel_comparison(mixer):
    """Create a color wheel showing pigment vs RGB mixing."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw=dict(projection="polar")
    )

    primaries = [
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (255, 0, 255),
    ]

    n_steps = 60
    angles = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    width = 2 * np.pi / n_steps

    for i, angle in enumerate(angles):
        t = i / n_steps
        idx1 = int(t * len(primaries)) % len(primaries)
        idx2 = (idx1 + 1) % len(primaries)
        local_t = (t * len(primaries)) % 1.0

        rgb_color = np.array(rgb_lerp(primaries[idx1], primaries[idx2], local_t)) / 255.0
        pigment_color = (
            np.array(mixer.lerp(*primaries[idx1], *primaries[idx2], local_t)) / 255.0
        )

        ax1.bar(angle, 1, width=width, bottom=0.3, color=rgb_color, edgecolor="none")
        ax2.bar(
            angle, 1, width=width, bottom=0.3, color=pigment_color, edgecolor="none"
        )

    for ax, title in [(ax1, "RGB Mixing\n(Desaturated)"), (ax2, "Pigment Mixing\n(Vibrant)")]:
        ax.set_ylim(0, 1.3)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig("color_wheel.png", dpi=200, bbox_inches="tight")
    print("Saved color wheel to color_wheel.png")


if __name__ == "__main__":
    print("Creating visual comparisons...")
    plot_comparison()
    print("\nDone!")
