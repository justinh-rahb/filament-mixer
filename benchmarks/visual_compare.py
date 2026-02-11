#!/usr/bin/env python3
"""
Visual Benchmark: Generate comparison grid images

Creates a visual grid comparing color mixing results across all methods:
- RGB Linear Interpolation
- FilamentMixer (K-M theory)
- FastLUTMixer (precomputed)
- Mixbox (if available)

Usage:
    python benchmarks/visual_compare.py
    python benchmarks/visual_compare.py --output my_comparison.png
"""

import argparse
import numpy as np
from pathlib import Path

from filament_mixer import FilamentMixer, CMYW_PALETTE

# Try imports
try:
    from filament_mixer import FastLUTMixer
    HAS_LUT = True
except ImportError:
    HAS_LUT = False

try:
    from filament_mixer import PolyMixer
    _poly_mixer = PolyMixer.from_cache("models")
    HAS_POLY = True
except (ImportError, FileNotFoundError):
    HAS_POLY = False
    _poly_mixer = None

try:
    from filament_mixer import GPMixer
    HAS_GP = True
except ImportError:
    HAS_GP = False

try:
    import mixbox
    HAS_MIXBOX = True
except ImportError:
    HAS_MIXBOX = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Error: PIL/Pillow required for visual comparison")
    print("Install with: pip install -e '.[lut]'")
    exit(1)


# Test color pairs (same as Mixbox benchmark)
COLOR_PAIRS = [
    ("Blue + Yellow", (0, 33, 133), (252, 211, 0)),
    ("Red + Blue", (255, 39, 2), (0, 33, 133)),
    ("Red + Yellow", (255, 39, 2), (252, 211, 0)),
    ("Magenta + Yellow", (128, 2, 46), (252, 211, 0)),
    ("Blue + White", (0, 33, 133), (255, 255, 255)),
    ("Red + White", (255, 39, 2), (255, 255, 255)),
    ("Green + Magenta", (0, 60, 50), (128, 2, 46)),
]


def rgb_lerp(c1, c2, t):
    """Naive RGB linear interpolation."""
    return tuple(int((1 - t) * c1[i] + t * c2[i]) for i in range(3))


def create_color_swatch(color, width, height):
    """Create a solid color image."""
    img = Image.new('RGB', (width, height), color)
    return img


def add_text_label(img, text, position="bottom", bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Add text label to image."""
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Get text bbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position text
    if position == "bottom":
        x = (img.width - text_width) // 2
        y = img.height - text_height - 10
    elif position == "top":
        x = (img.width - text_width) // 2
        y = 10
    else:
        x, y = position
    
    # Draw background rectangle
    padding = 4
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=bg_color
    )
    
    # Draw text
    draw.text((x, y), text, fill=text_color, font=font)
    
    return img


def create_gradient_strip(c1, c2, width, height, steps=5):
    """Create a gradient strip showing mixing steps."""
    img = Image.new('RGB', (width, height))
    step_width = width // steps
    
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0.5
        color = tuple(int((1 - t) * c1[j] + t * c2[j]) for j in range(3))
        
        for x in range(i * step_width, min((i + 1) * step_width, width)):
            for y in range(height):
                img.putpixel((x, y), color)
    
    return img


def generate_comparison_grid(output_path="benchmarks/visual_comparison.png", swatch_size=120, lut_resolution=256):
    """Generate a comprehensive visual comparison grid."""
    
    # Initialize mixers
    fm_mixer = FilamentMixer(CMYW_PALETTE)
    
    lut_mixer = None
    if HAS_LUT:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=lut_resolution)
            print(f"✓ Using FastLUTMixer ({lut_resolution}³ LUT)")
        except FileNotFoundError:
            print(f"⚠ No {lut_resolution}³ LUT found (run: python scripts/generate_lut.py --resolution {lut_resolution})")
            HAS_LUT_LOADED = False
        else:
            HAS_LUT_LOADED = True
    else:
        HAS_LUT_LOADED = False
    
    if HAS_POLY:
        print("✓ Using PolyMixer")
    
    gp_mixer = None
    if HAS_GP:
        try:
            gp_mixer = GPMixer.from_cache("models")
            print("✓ Using GPMixer (Experiment C)")
            HAS_GP_LOADED = True
        except FileNotFoundError:
            print("⚠ GPMixer model not found (run: python scripts/train_gp_model.py)")
            HAS_GP_LOADED = False
    else:
        HAS_GP_LOADED = False
    
    if HAS_MIXBOX:
        print("✓ Using Mixbox for comparison")
    
    # Calculate grid dimensions
    num_pairs = len(COLOR_PAIRS)
    num_methods = 2 + (1 if HAS_LUT_LOADED else 0) + (1 if HAS_POLY else 0) + (1 if HAS_GP_LOADED else 0) + (1 if HAS_MIXBOX else 0)
    
    # Layout: Each row shows: C1 | C2 | RGB | FM | LUT? | Poly? | GP? | Mixbox?
    cell_width = swatch_size
    cell_height = swatch_size
    margin = 10
    label_height = 30
    header_height = 40
    
    grid_width = (2 + num_methods) * cell_width + (1 + num_methods) * margin
    grid_height = header_height + num_pairs * (cell_height + label_height + margin) + margin
    
    # Create canvas
    canvas = Image.new('RGB', (grid_width, grid_height), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    
    # Try to load font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = header_font = label_font = ImageFont.load_default()
    
    # Draw title
    title = "Color Mixing Comparison"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((grid_width - title_width) // 2, 10), title, fill=(0, 0, 0), font=title_font)
    
    # Draw column headers
    headers = ["Color 1", "Color 2", "RGB Lerp", "FilamentMixer"]
    if HAS_LUT_LOADED:
        headers.append("FastLUT")
    if HAS_POLY:
        headers.append("PolyMixer")
    if HAS_GP_LOADED:
        headers.append("GPMixer")
    if HAS_MIXBOX:
        headers.append("Mixbox")
    
    y_header = header_height
    for col_idx, header_text in enumerate(headers):
        x = margin + col_idx * (cell_width + margin)
        bbox = draw.textbbox((0, 0), header_text, font=header_font)
        text_width = bbox[2] - bbox[0]
        draw.text((x + (cell_width - text_width) // 2, y_header), 
                 header_text, fill=(60, 60, 60), font=header_font)
    
    # Process each color pair
    y_offset = header_height + header_height
    
    for pair_idx, (name, c1, c2) in enumerate(COLOR_PAIRS):
        x_offset = margin
        
        # Color 1
        swatch1 = create_color_swatch(c1, cell_width, cell_height)
        canvas.paste(swatch1, (x_offset, y_offset))
        x_offset += cell_width + margin
        
        # Color 2
        swatch2 = create_color_swatch(c2, cell_width, cell_height)
        canvas.paste(swatch2, (x_offset, y_offset))
        x_offset += cell_width + margin
        
        # RGB Lerp
        rgb_result = rgb_lerp(c1, c2, 0.5)
        swatch_rgb = create_color_swatch(rgb_result, cell_width, cell_height)
        canvas.paste(swatch_rgb, (x_offset, y_offset))
        x_offset += cell_width + margin
        
        # FilamentMixer
        fm_result = fm_mixer.lerp(*c1, *c2, 0.5)
        swatch_fm = create_color_swatch(fm_result, cell_width, cell_height)
        canvas.paste(swatch_fm, (x_offset, y_offset))
        x_offset += cell_width + margin
        
        # FastLUT
        if HAS_LUT_LOADED:
            lut_result = lut_mixer.lerp(*c1, *c2, 0.5)
            swatch_lut = create_color_swatch(lut_result, cell_width, cell_height)
            canvas.paste(swatch_lut, (x_offset, y_offset))
            x_offset += cell_width + margin
        
        # PolyMixer
        if HAS_POLY:
            poly_result = _poly_mixer.lerp(*c1, *c2, 0.5)
            swatch_poly = create_color_swatch(poly_result, cell_width, cell_height)
            canvas.paste(swatch_poly, (x_offset, y_offset))
            x_offset += cell_width + margin
        
        # GPMixer
        if HAS_GP_LOADED:
            gp_result = gp_mixer.lerp(*c1, *c2, 0.5)
            swatch_gp = create_color_swatch(gp_result, cell_width, cell_height)
            canvas.paste(swatch_gp, (x_offset, y_offset))
            x_offset += cell_width + margin
        
        # Mixbox
        if HAS_MIXBOX:
            mixbox_result = mixbox.lerp(c1, c2, 0.5)
            swatch_mixbox = create_color_swatch(mixbox_result, cell_width, cell_height)
            canvas.paste(swatch_mixbox, (x_offset, y_offset))
            x_offset += cell_width + margin
        
        # Draw row label
        label_y = y_offset + cell_height + 5
        bbox = draw.textbbox((0, 0), name, font=label_font)
        text_width = bbox[2] - bbox[0]
        draw.text((margin + (2 * cell_width + margin - text_width) // 2, label_y), 
                 name, fill=(0, 0, 0), font=label_font)
        
        y_offset += cell_height + label_height + margin
    
    # Save image
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    print(f"\n✓ Saved visual comparison to: {output_file}")
    print(f"  Grid size: {grid_width}x{grid_height} pixels")
    print(f"  Test pairs: {num_pairs}")
    print(f"  Methods: {num_methods}")
    
    return output_file


def generate_gradient_comparison(output_path="benchmarks/gradient_comparison.png", lut_resolution=256):
    """Generate gradient strips showing smooth color transitions."""
    
    # Initialize mixers
    fm_mixer = FilamentMixer(CMYW_PALETTE)
    
    lut_mixer = None
    if HAS_LUT:
        try:
            lut_mixer = FastLUTMixer.from_cache("lut_cache", resolution=lut_resolution)
            HAS_LUT_LOADED = True
        except FileNotFoundError:
            HAS_LUT_LOADED = False
    else:
        HAS_LUT_LOADED = False
    
    gp_mixer = None
    if HAS_GP:
        try:
            gp_mixer = GPMixer.from_cache("models")
            HAS_GP_LOADED = True
        except FileNotFoundError:
            HAS_GP_LOADED = False
    else:
        HAS_GP_LOADED = False
    
    # Select a few interesting pairs
    selected_pairs = [
        ("Blue + Yellow → Green", (0, 33, 133), (252, 211, 0)),
        ("Red + Blue → Purple", (255, 39, 2), (0, 33, 133)),
        ("Red + Yellow → Orange", (255, 39, 2), (252, 211, 0)),
    ]
    
    strip_width = 800
    strip_height = 80
    margin = 20
    label_height = 40
    steps = 9  # Number of gradient steps
    
    num_methods = 2 + (1 if HAS_LUT_LOADED else 0) + (1 if HAS_POLY else 0) + (1 if HAS_GP_LOADED else 0) + (1 if HAS_MIXBOX else 0)
    canvas_height = len(selected_pairs) * (num_methods * (strip_height + 5) + label_height + margin) + margin
    
    canvas = Image.new('RGB', (strip_width + 2 * margin, canvas_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    
    # Load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = label_font = ImageFont.load_default()
    
    y_offset = margin
    
    for name, c1, c2 in selected_pairs:
        # Draw pair label
        draw.text((margin, y_offset), name, fill=(0, 0, 0), font=font)
        y_offset += label_height
        
        # RGB gradient
        rgb_strip = Image.new('RGB', (strip_width, strip_height))
        step_width = strip_width // steps
        for i in range(steps):
            t = i / (steps - 1)
            color = rgb_lerp(c1, c2, t)
            for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                for y in range(strip_height):
                    rgb_strip.putpixel((x, y), color)
        
        draw.text((margin, y_offset), "RGB Lerp", fill=(80, 80, 80), font=label_font)
        canvas.paste(rgb_strip, (margin, y_offset + 20))
        y_offset += strip_height + 5
        
        # FilamentMixer gradient
        fm_strip = Image.new('RGB', (strip_width, strip_height))
        for i in range(steps):
            t = i / (steps - 1)
            color = fm_mixer.lerp(*c1, *c2, t)
            for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                for y in range(strip_height):
                    fm_strip.putpixel((x, y), color)
        
        draw.text((margin, y_offset), "FilamentMixer", fill=(80, 80, 80), font=label_font)
        canvas.paste(fm_strip, (margin, y_offset + 20))
        y_offset += strip_height + 5
        
        # LUT gradient
        if HAS_LUT_LOADED:
            lut_strip = Image.new('RGB', (strip_width, strip_height))
            for i in range(steps):
                t = i / (steps - 1)
                color = lut_mixer.lerp(*c1, *c2, t)
                for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                    for y in range(strip_height):
                        lut_strip.putpixel((x, y), color)
            
            draw.text((margin, y_offset), "FastLUT", fill=(80, 80, 80), font=label_font)
            canvas.paste(lut_strip, (margin, y_offset + 20))
            y_offset += strip_height + 5
        
        # PolyMixer gradient
        if HAS_POLY:
            poly_strip = Image.new('RGB', (strip_width, strip_height))
            for i in range(steps):
                t = i / (steps - 1)
                color = _poly_mixer.lerp(*c1, *c2, t)
                for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                    for y in range(strip_height):
                        poly_strip.putpixel((x, y), color)
            
            draw.text((margin, y_offset), "PolyMixer", fill=(80, 80, 80), font=label_font)
            canvas.paste(poly_strip, (margin, y_offset + 20))
            y_offset += strip_height + 5
        
        # GP gradient
        if HAS_GP_LOADED:
            gp_strip = Image.new('RGB', (strip_width, strip_height))
            for i in range(steps):
                t = i / (steps - 1)
                color = gp_mixer.lerp(*c1, *c2, t)
                for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                    for y in range(strip_height):
                        gp_strip.putpixel((x, y), color)
            
            draw.text((margin, y_offset), "GPMixer", fill=(80, 80, 80), font=label_font)
            canvas.paste(gp_strip, (margin, y_offset + 20))
            y_offset += strip_height + 5
        
        # Mixbox gradient
        if HAS_MIXBOX:
            mixbox_strip = Image.new('RGB', (strip_width, strip_height))
            for i in range(steps):
                t = i / (steps - 1)
                color = mixbox.lerp(c1, c2, t)
                for x in range(i * step_width, min((i + 1) * step_width, strip_width)):
                    for y in range(strip_height):
                        mixbox_strip.putpixel((x, y), color)
            
            draw.text((margin, y_offset), "Mixbox", fill=(80, 80, 80), font=label_font)
            canvas.paste(mixbox_strip, (margin, y_offset + 20))
            y_offset += strip_height + 5
        
        y_offset += margin
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file)
    print(f"✓ Saved gradient comparison to: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate visual color mixing comparisons")
    parser.add_argument(
        "--grid",
        type=str,
        default="benchmarks/visual_comparison.png",
        help="Output path for swatch grid comparison"
    )
    parser.add_argument(
        "--gradient",
        type=str,
        default="benchmarks/gradient_comparison.png",
        help="Output path for gradient comparison"
    )
    parser.add_argument(
        "--swatch-size",
        type=int,
        default=120,
        help="Size of color swatches in pixels"
    )
    parser.add_argument(
        "--skip-gradient",
        action="store_true",
        help="Skip gradient comparison"
    )
    parser.add_argument(
        "--lut-resolution",
        type=int,
        default=256,
        choices=[64, 256],
        help="LUT resolution to use (64 or 256)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Visual Color Mixing Comparison")
    print("=" * 70)
    print()
    
    # Generate swatch grid
    print("Generating swatch grid comparison...")
    generate_comparison_grid(args.grid, args.swatch_size, lut_resolution=args.lut_resolution)
    
    # Generate gradient comparison
    if not args.skip_gradient:
        print("\nGenerating gradient comparison...")
        generate_gradient_comparison(args.gradient, lut_resolution=args.lut_resolution)
    
    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
