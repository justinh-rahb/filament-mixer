/*
 * FilamentMixer — Header-only C++ pigment color mixer
 *
 * Drop-in replacement for mixbox. Uses a degree-4 polynomial regression
 * trained on Mixbox ground truth (Mean Delta-E ~2.07).
 *
 * Usage:
 *   #include "filament_mixer.h"
 *
 *   unsigned char r, g, b;
 *   filament_mixer::lerp(0, 33, 133,  252, 211, 0,  0.5f, &r, &g, &b);
 *   // r=47, g=141, b=56  (blue + yellow → green)
 *
 * No dependencies beyond the C++ standard library.
 *
 * License: Same as the parent filament-mixer project.
 */

#ifndef FILAMENT_MIXER_H
#define FILAMENT_MIXER_H

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace filament_mixer {
namespace detail {

#include "filament_mixer_data.inc"

inline void compute_poly_features(const double x[7], double out[330]) {
    for (int i = 0; i < N_FEATURES; ++i) {
        double val = 1.0;
        for (int j = 0; j < N_INPUTS; ++j) {
            if (POWERS[i][j] != 0) {
                double base = x[j];
                int exp = POWERS[i][j];
                // Fast integer exponentiation (max exp = 4)
                double p = 1.0;
                for (int e = 0; e < exp; ++e)
                    p *= base;
                val *= p;
            }
        }
        out[i] = val;
    }
}

} // namespace detail

struct RGB {
    unsigned char r, g, b;
};

/**
 * Mix two RGB colors using polynomial pigment mixing.
 *
 * This is a drop-in replacement for mixbox_lerp.
 *
 * @param r1,g1,b1  First color (0-255)
 * @param r2,g2,b2  Second color (0-255)
 * @param t         Mixing ratio: 0.0 = all color1, 1.0 = all color2
 * @param out_r,out_g,out_b  Output color (0-255)
 */
inline void lerp(unsigned char r1, unsigned char g1, unsigned char b1,
                 unsigned char r2, unsigned char g2, unsigned char b2,
                 float t,
                 unsigned char* out_r, unsigned char* out_g, unsigned char* out_b) {
    // Clamp t
    if (t <= 0.0f) {
        *out_r = r1; *out_g = g1; *out_b = b1;
        return;
    }
    if (t >= 1.0f) {
        *out_r = r2; *out_g = g2; *out_b = b2;
        return;
    }

    double x[7] = {
        static_cast<double>(r1), static_cast<double>(g1), static_cast<double>(b1),
        static_cast<double>(r2), static_cast<double>(g2), static_cast<double>(b2),
        static_cast<double>(t)
    };

    double features[330];
    detail::compute_poly_features(x, features);

    // Dot product: features @ COEF + INTERCEPT
    for (int c = 0; c < 3; ++c) {
        double sum = detail::INTERCEPT[c];
        for (int i = 0; i < detail::N_FEATURES; ++i) {
            sum += features[i] * detail::COEF[i][c];
        }
        // Clamp to [0, 255] and truncate (matches numpy astype(int) behavior)
        int val = static_cast<int>(sum);
        if (val < 0) val = 0;
        if (val > 255) val = 255;

        if (c == 0) *out_r = static_cast<unsigned char>(val);
        else if (c == 1) *out_g = static_cast<unsigned char>(val);
        else *out_b = static_cast<unsigned char>(val);
    }
}

/**
 * Convenience overload returning an RGB struct.
 */
inline RGB lerp(unsigned char r1, unsigned char g1, unsigned char b1,
                unsigned char r2, unsigned char g2, unsigned char b2,
                float t) {
    RGB result;
    lerp(r1, g1, b1, r2, g2, b2, t, &result.r, &result.g, &result.b);
    return result;
}

} // namespace filament_mixer

#endif // FILAMENT_MIXER_H
