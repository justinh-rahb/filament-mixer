#include "filament_mixer.h"
#include <cstdio>

int main() {
    // Test cases verified against Python PolyMixer output
    struct TestCase {
        unsigned char r1, g1, b1, r2, g2, b2;
        float t;
        unsigned char exp_r, exp_g, exp_b;
        const char* label;
    };

    TestCase tests[] = {
        {  0,  33, 133, 252, 211,   0, 0.50f,  47, 141,  56, "Blue + Yellow (the green problem)"},
        {255,   0,   0, 255, 255, 255, 0.50f, 255, 101, 136, "Red + White"},
        {  0,   0, 255, 255, 255,   0, 0.25f,  28,  80, 171, "Blue + Yellow t=0.25"},
        {255,   0,   0,   0,   0, 255, 0.50f, 111,   0, 110, "Red + Blue"},
        {128, 128, 128, 128, 128, 128, 0.50f, 129, 128, 125, "Gray + Gray"},
    };

    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;

    printf("FilamentMixer C++ — Test Results\n");
    printf("================================\n\n");

    for (int i = 0; i < n_tests; ++i) {
        auto& tc = tests[i];
        auto result = filament_mixer::lerp(
            tc.r1, tc.g1, tc.b1,
            tc.r2, tc.g2, tc.b2,
            tc.t
        );

        bool ok = (result.r == tc.exp_r && result.g == tc.exp_g && result.b == tc.exp_b);
        passed += ok;

        printf("  %s\n", tc.label);
        printf("    lerp((%3d,%3d,%3d), (%3d,%3d,%3d), %.2f)\n",
               tc.r1, tc.g1, tc.b1, tc.r2, tc.g2, tc.b2, tc.t);
        printf("    Got:      (%3d, %3d, %3d)\n", result.r, result.g, result.b);
        printf("    Expected: (%3d, %3d, %3d)  %s\n\n",
               tc.exp_r, tc.exp_g, tc.exp_b, ok ? "PASS" : "FAIL");
    }

    // Edge cases: t=0 and t=1 should return exact input colors
    auto edge0 = filament_mixer::lerp(255, 0, 0, 0, 0, 255, 0.0f);
    auto edge1 = filament_mixer::lerp(255, 0, 0, 0, 0, 255, 1.0f);

    bool e0_ok = (edge0.r == 255 && edge0.g == 0 && edge0.b == 0);
    bool e1_ok = (edge1.r == 0 && edge1.g == 0 && edge1.b == 255);
    passed += e0_ok + e1_ok;
    n_tests += 2;

    printf("  Edge: t=0 -> (%d,%d,%d) %s\n", edge0.r, edge0.g, edge0.b, e0_ok ? "PASS" : "FAIL");
    printf("  Edge: t=1 -> (%d,%d,%d) %s\n\n", edge1.r, edge1.g, edge1.b, e1_ok ? "PASS" : "FAIL");

    printf("Result: %d/%d passed\n", passed, n_tests);
    return (passed == n_tests) ? 0 : 1;
}
