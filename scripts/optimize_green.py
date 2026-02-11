
import numpy as np
import argparse
import time
import mixbox
from scipy.optimize import minimize
from filament_mixer import FilamentMixer, Pigment, CIE_WAVELENGTHS

# ---------------------------------------------------------------------------
# 1. Pigment Factory (Parametric)
# ---------------------------------------------------------------------------

def gaussian_peak(wavelengths, center, width, height):
    return height * np.exp(-((wavelengths - center) / width) ** 2)

def make_white(params):
    # params: [K_val, S_val]
    K = np.ones(len(CIE_WAVELENGTHS)) * params[0]
    S = np.ones(len(CIE_WAVELENGTHS)) * params[1]
    return Pigment("White", K, S)

def make_cyan(params):
    # params: [r_center, r_width, r_height, o_center, o_width, o_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2])
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5])
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Cyan", K, S)

def make_magenta(params):
    # params: [g1_center, g1_width, g1_height, g2_center, g2_width, g2_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.03
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2])
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5])
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Magenta", K, S)

def make_yellow(params):
    # params: [b_center, b_width, b_height, v_center, v_width, v_height]
    K = np.ones(len(CIE_WAVELENGTHS)) * 0.02
    K += gaussian_peak(CIE_WAVELENGTHS, params[0], params[1], params[2])
    K += gaussian_peak(CIE_WAVELENGTHS, params[3], params[4], params[5])
    S = np.ones(len(CIE_WAVELENGTHS)) * 0.5
    return Pigment("Yellow", K, S)

# ---------------------------------------------------------------------------
# 2. Optimization Logic
# ---------------------------------------------------------------------------

def rgb_to_lab(rgb):
    from skimage import color
    return color.rgb2lab(np.array(rgb).reshape(1, 1, 3) / 255.0).flatten()

def loss_function(params, training_data):
    # Unpack params
    c_params = params[0:6]
    m_params = params[6:12]
    y_params = params[12:18]
    w_params = params[18:20]
    
    palette = [
        make_cyan(c_params),
        make_magenta(m_params),
        make_yellow(y_params),
        make_white(w_params)
    ]
    
    mixer = FilamentMixer(palette)
    # Note: We do NOT set mixer.unmixer = None here, as it breaks lerp()
    
    total_error = 0.0
    total_weight = 0.0
    
    for sample in training_data:
        c1, c2, t = sample['c1'], sample['c2'], sample['t']
        target = sample['target']
        weight = sample['weight']
        
        try:
            predicted = mixer.lerp(*c1, *c2, t)
            
            # Weighted CIE76 Delta-E
            lab_p = rgb_to_lab(predicted)
            lab_t = rgb_to_lab(target)
            
            dE = np.sqrt(np.sum((lab_p - lab_t) ** 2))
            
            total_error += dE * weight
            total_weight += weight
            
        except Exception as e:
            # Only print first few errors to avoid spam
            if total_weight == 0: 
                print(f"Error in loss_function: {e}")
            total_error += 100 * weight # Penalize failure
            total_weight += weight

    return total_error / total_weight if total_weight > 0 else 0

# Initial "Best Known" Parameters (Automated Result from previous run)
# Cyan (6), Magenta (6), Yellow (6), White (2) = 20 params
INITIAL_PARAMS = [
    640, 45, 2.5, 600, 20, 1.5,   # Cyan
    530, 20, 3.0, 555, 20, 2.5,   # Magenta
    450, 45, 2.5, 400, 40, 2.0,   # Yellow
    0.0989, 0.5                   # White (Optimized)
]

# Bounds
BOUNDS = [
    (600, 700), (10, 100), (0.1, 5.0), (550, 650), (10, 100), (0.1, 5.0), # Cyan
    (500, 560), (10, 100), (0.1, 5.0), (500, 600), (10, 100), (0.1, 5.0), # Magenta
    (400, 500), (10, 100), (0.1, 5.0), (380, 450), (10, 100), (0.1, 5.0), # Yellow
    (0.001, 1.0), (0.01, 5.0)                                             # White
]

def generate_training_data(n_random=20):
    data = []
    
    # 1. Critical Pair: Blue + Yellow (The Green Problem)
    # Weight = 50.0 (Massive priority)
    data.append({
        'c1': (0, 33, 133), 
        'c2': (252, 211, 0), 
        't': 0.5, 
        'target': mixbox.lerp((0, 33, 133), (252, 211, 0), 0.5),
        'weight': 50.0
    })
    
    # 2. Critical Pair: Red + Cyan (Purple)
    data.append({
        'c1': (255, 0, 0), 
        'c2': (0, 255, 255), 
        't': 0.5, 
        'target': mixbox.lerp((255, 0, 0), (0, 255, 255), 0.5),
        'weight': 10.0
    })

    # 3. Random background samples (Weight 1.0)
    for _ in range(n_random):
        c1 = tuple(np.random.randint(0, 256, 3))
        c2 = tuple(np.random.randint(0, 256, 3))
        data.append({
            'c1': c1, 'c2': c2, 't': 0.5,
            'target': mixbox.lerp(c1, c2, 0.5),
            'weight': 1.0
        })
        
    return data

iteration_count = 0
training_data = None

def print_progress(xk):
    global iteration_count, training_data
    iteration_count += 1
    # Check Green specifically
    loss = loss_function(xk, training_data)
    
    # Check Blue+Yellow mix specifically
    # Reconstruct mixer to check
    c_params = xk[0:6]
    m_params = xk[6:12]
    y_params = xk[12:18]
    w_params = xk[18:20]
    palette = [make_cyan(c_params), make_magenta(m_params), make_yellow(y_params), make_white(w_params)]
    mixer = FilamentMixer(palette)
    green_mix = mixer.lerp(0, 33, 133, 252, 211, 0, 0.5)
    
    print(f"Iter {iteration_count}: Loss={loss:.4f} | B+Y Result={green_mix}")

def main():
    global training_data
    print("Initializing Experiment B: Green-Targeted Optimization...")
    training_data = generate_training_data(10) # 10 random + 2 critical
    
    print("Starting optimization (L-BFGS-B)...")
    
    # 1. Print Initial State
    print("\n--- Initial State ---")
    print_progress(INITIAL_PARAMS)
    
    start_time = time.time()
    result = minimize(
        loss_function,
        INITIAL_PARAMS,
        args=(training_data,),
        method='L-BFGS-B',
        bounds=BOUNDS,
        callback=print_progress,
        options={'maxiter': 20, 'disp': True, 'eps': 1e-4}
    )
    
    print(f"\nOptimization Complete in {time.time() - start_time:.2f}s")
    print(f"Final Success: {result.success}")
    print(f"Message: {result.message}")
    
    # 2. Print Final State
    print("\n--- Final State ---")
    print_progress(result.x)
    print(f"\nOptimized Parameters: {list(result.x)}")
    
if __name__ == "__main__":
    main()
