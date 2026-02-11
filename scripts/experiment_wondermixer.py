"""
Experiment E: Wondermixer - Pre-fitted Total Degree 3 Polynomial

A lightweight cubic polynomial approach using hypercubic fitting.
Uses a full monomial feature map up to total degree 3 for the 7-dimensional
input space [(R1, G1, B1), (R2, G2, B2), T].

Credit: Wombley
Approximate fit to MIXBOX by Sarka Sochorova and Ondrej Jamriska
  Website: https://scrtwpns.com/mixbox/
  GitHub: https://github.com/scrtwpns/mixbox

Advantages:
- Only requires numpy (no sklearn/scipy)
- Pre-trained coefficients (no training needed)
- Fast inference
- Good accuracy across most of the range

Known limitations:
- Can be slightly off at extreme ratios (>0.9, <0.1) when mixing with white
"""

import numpy as np
import time
import mixbox
from skimage import color


def poly_features_td3(Z: np.ndarray) -> np.ndarray:
    """
    Full monomial feature map up to total degree 3 for N-by-d input Z.
    Canonical ordering:
      degree 0: 1
      degree 1: z_i
      degree 2: z_i z_j  (i <= j)
      degree 3: z_i z_j z_k (i <= j <= k)

    Z: shape (N, d) or (d,) for a single sample
    Returns Phi: shape (N, n_terms) where n_terms = comb(d+3, 3)
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[None, :]
    N, d = Z.shape

    # number of terms = C(d+3, 3)
    n_terms = (d + 3) * (d + 2) * (d + 1) // 6
    Phi = np.empty((N, n_terms), dtype=float)

    idx = 0

    # Degree 0
    Phi[:, idx] = 1.0
    idx += 1

    # Degree 1
    Phi[:, idx:idx + d] = Z
    idx += d

    # Degree 2: i <= j
    for i in range(d):
        zi = Z[:, i]
        for j in range(i, d):
            Phi[:, idx] = zi * Z[:, j]
            idx += 1

    # Degree 3: i <= j <= k
    for i in range(d):
        zi = Z[:, i]
        for j in range(i, d):
            zij = zi * Z[:, j]
            for k in range(j, d):
                Phi[:, idx] = zij * Z[:, k]
                idx += 1

    if idx != n_terms:
        raise RuntimeError(f"Feature count mismatch: built {idx}, expected {n_terms}")

    return Phi


def load_beta() -> np.ndarray:
    """
    Returns beta as shape (120, 3) for 7D hypercubic (total degree <= 3).

    These are the [R; G; B] hypercubic fitting parameters for the 7-dimensional
    space [R1, G1, B1, R2, G2, B2, T], approximate fit to MIXBOX by Sarka Sochorova and Ondrej Jamriska:
      Website -- https://scrtwpns.com/mixbox/
      github -- https://github.com/scrtwpns/mixbox

    Each column corresponds to each of the 120 R, G and B coefficients.
    """
    b = np.array([
        [0.505650869791944, 0.500924112979576, 0.480577124594652],
        [0.250238771490529, -0.00488249500483106, -0.000292577710437636],
        [-0.00186894180608081, 0.261824557661893, -0.00603748549935585],
        [-0.00291003464642525, -0.00446009393922519, 0.257895399531569],
        [0.251992067696548, -0.00476023832109888, -0.00129025992434985],
        [-0.00102373867119944, 0.260942576001955, -0.00495080045481410],
        [-0.00332591872117008, -0.00487680972203331, 0.258799333739124],
        [0.000267804857135874, 0.000369035048702932, -0.000430727075001385],
        [-0.0161085278953243, 0.00323279969511681, 0.00441582542356634],
        [-0.0245729050592123, 0.0136701807307570, 0.00919605359037830],
        [0.0130883780570058, 0.00462250571229030, 0.0126602547301254],
        [0.0221685040620148, -0.00130618885113910, -0.000708432728067441],
        [0.0227972495031089, -0.0161542071678489, -0.00744960849106922],
        [-0.0146129187246876, -0.00469005484634862, -0.0176462208499584],
        [-0.247360384036530, 0.00115013018214376, 0.00132775904718365],
        [0.00209310152577968, -0.0178524239760968, 0.0150208302538610],
        [0.00970671212018641, -0.0210620707455843, 0.0101926728950161],
        [0.0222924220574049, -0.0157082043294293, -0.00701639670522254],
        [-0.00670962064149818, 0.0246628214997869, -0.00834749515182193],
        [-0.00711570499801769, 0.0217531829803320, -0.0105065264349561],
        [0.000262777593548254, -0.247920521874391, -0.000840306446118075],
        [0.00597360705327295, 0.0111206454809343, -0.0236247103347130],
        [-0.0149773274436049, -0.00479185905168239, -0.0171268869161580],
        [-0.00702471274951632, 0.0222752925460449, -0.0109514910196104],
        [-0.0131917061885572, -0.0258824340745628, 0.0541901948263466],
        [-0.00270391477143491, 0.00153791984618603, -0.245343313965382],
        [-0.0163161527443542, 0.00323105863745680, 0.00444015852902883],
        [-0.0249409056093806, 0.0133341637436519, 0.00984155941385434],
        [0.0129490759732505, 0.00483911544281080, 0.0125725097477904],
        [0.247240927061282, -0.00130666357537571, -0.00102308371049578],
        [0.00173051600289389, -0.0175132954189093, 0.0152282314908548],
        [0.00915561031831999, -0.0204815571804216, 0.0103549057577473],
        [-0.000400925868617201, 0.247923350963498, 0.000738245141772882],
        [0.00572126782437638, 0.0110039471836717, -0.0231633108774377],
        [0.00268595582532584, -0.00137760125159290, 0.245179452043062],
        [-4.89277941573016e-05, 0.00152417349561795, 0.0218483545467699],
        [0.00867903557934649, -0.00640997541601078, -0.0166617782835314],
        [0.0126129655188494, -0.00191234164665377, -0.00408844603274204],
        [0.00981580085042439, -0.0207031981592407, 0.00159162296002219],
        [-0.00667563960549800, 0.00484093905467446, 0.0142489341618179],
        [-0.00427210078153843, 0.00130965680232706, 0.00211945648880917],
        [-0.00130563098759029, 0.00929334065935962, -0.000102242239265027],
        [0.000300840559003422, 0.000100972249650500, -0.00159523034049590],
        [0.0153305520081890, -0.000806538944151330, -0.000399633583551340],
        [0.0102530108358341, -0.00311998644370665, -0.00434176268711205],
        [-0.00338007380612014, -0.00825561014741758, 0.000380398704937962],
        [-0.00933594918879589, 0.0101925099853651, -0.00390535665099917],
        [0.000160307758742811, 0.00380264566050350, 0.0116244254824966],
        [-9.00918834673194e-05, 0.00251842788224369, -0.00278240036944381],
        [-0.0275217430111522, -0.000708873437376815, 0.0112480068768416],
        [-0.00419966971353520, 0.0102480310950305, -0.00362833343762153],
        [-0.000656620751038066, -0.00114421480664705, -0.00195995779760673],
        [0.00701814570430622, 0.00333822944379730, -0.00250836588883003],
        [-0.00124592642872632, 0.000425519581373264, 0.00355571528331818],
        [-0.00648641680307997, 0.00457621658249891, 0.0146691586925890],
        [-0.00282428866643291, -0.00822220096806289, 0.000727511636887823],
        [-0.00502985545383594, 0.0106981890700083, -0.00345031636850162],
        [8.05591101191423e-05, 0.000247134418955104, 3.29174171042099e-05],
        [-0.00557644411465746, -0.00324740662632069, 0.000522780108216342],
        [-0.0102235607872258, 0.000300489595318753, -0.00265896929908900],
        [0.00107527856404291, 0.000479128998749812, -0.00277042324167493],
        [0.0195943190773994, 0.000456545743494954, -0.00406135845434441],
        [-0.00247350298000291, 7.40081557434467e-05, 0.000904253727795105],
        [-0.00314003665105625, 0.00837326435159818, 0.00219009420439416],
        [-0.00510102386335844, 0.00610914191516321, 0.00360687389523970],
        [-0.00788277341099206, 0.0258339936569502, -0.0237632613613611],
        [-0.00665414646596563, -0.00308793687884154, 0.000666615005811301],
        [0.00616398002855882, -0.0111918671603941, 0.000141603713695473],
        [-0.00136062653794897, -0.0109392954073518, 0.0125381615328413],
        [0.000847081536907015, -0.00249893576682105, 0.00143649265934990],
        [-0.00557587793895177, 0.00280085293557953, -0.0134442747960756],
        [-0.00936627703064292, 0.000525223218695932, -0.00234320736161009],
        [0.00823923900161096, -0.0114941198995554, 0.00649285716379649],
        [-0.00234293122966816, -0.0128735359203162, 0.0175191782214614],
        [0.000122273440498401, -0.00291215257827156, 0.00273422319079156],
        [-0.00273290512375276, -0.000582739082145268, 0.00215575539272608],
        [-0.00845147726486643, 0.0100473404604294, -0.00407246846681943],
        [-0.00118822056208743, -0.000724603770213656, -0.00246440445557021],
        [-0.000698400610113906, -0.00130827571508250, 0.00270622501549299],
        [0.00621515572920421, -0.0107169909123842, -0.000549922895298204],
        [0.00772131293783984, -0.0106790276543973, 0.00546851757014452],
        [-0.000535000941941677, 0.000787877390262898, -0.000130969260747114],
        [0.00310581647357724, 0.00292140347698507, -0.000514423762989031],
        [-0.000604558904703217, 0.000971178766504083, 0.00256218883947234],
        [0.000806219739804866, -0.0129827647625922, 0.0101177735961012],
        [0.00285380605268331, -0.00308296691809302, 0.00858225278608351],
        [0.0192952740390909, 0.000318181117977998, -0.00324120845485394],
        [0.00257579311529095, 0.00324838136475642, -0.000614141789491056],
        [-0.00167751680139074, 0.00931234387019601, -0.0165792067771108],
        [0.00174959782898140, 0.00131584687056572, -0.00358055948711220],
        [-0.00125371447529379, 0.00909309095273496, 0.000390833337090701],
        [0.000209665958916591, 0.00392454756000299, 0.0102504506284882],
        [0.00806351192955983, 0.00296648758192920, -0.00283198173686435],
        [0.00240151280079882, -0.000406389397310547, -0.00173175341201101],
        [-0.00148979753523543, -0.0117878808602999, 0.0126069362216734],
        [-0.00229912478591901, -0.0123643769572274, 0.0181995100970968],
        [0.000660171759784872, -0.000525056262884013, -0.00282115637933745],
        [-0.00158812835551211, 0.00969531301433179, -0.0167867721467253],
        [0.000264036994603328, -0.000417692268721275, 0.000426553959952494],
        [0.00182282139044119, 0.00218005246711435, -0.00404258789510149],
        [0.00806964481629853, -0.00705956386739597, -0.0157474317126586],
        [0.0117105611953397, -0.00132542586866922, -0.00423165422354869],
        [0.0105467360313613, -0.0209330275380502, 0.000788482430512412],
        [-0.000503389888833197, -0.000329062709077785, 0.00147869115863279],
        [0.0144798078226888, -0.000672548318923826, -0.000698176663838788],
        [0.0116005361515222, -0.00360756377515016, -0.00438625129695535],
        [-0.000286835757491530, -0.00178599920178871, 0.00257581264995833],
        [-0.0278919758784198, -0.000809190774697984, 0.0107313582461439],
        [0.00152726475724449, -0.000380695024681651, -0.00214825011317872],
        [-0.00481902481513825, 0.00899808292565524, 0.00364330172516394],
        [-0.00492364402385906, 0.00606436238026575, 0.00240687494101083],
        [-0.00660645409068592, 0.0245207751486620, -0.0241863509903595],
        [-0.000116763559010185, 0.00160296085974737, -0.00103071855609400],
        [-0.00529355506729314, 0.00246230289902102, -0.0136719014052472],
        [9.25643887981376e-05, 0.00300317223514024, -0.00258907084455951],
        [8.72471694750693e-05, -0.0126618615828678, 0.00885528432155944],
        [0.00279926750920340, -0.00246112291843894, 0.00862075872553931],
        [-0.00225614984760094, -0.00102153766982813, 0.00352624174802616],
        [0.00179480543414148, 0.00207285734855312, -0.00489655055174879],
        [-0.000429750466849352, -0.000156419532119436, 0.000399511281688479],
    ], dtype=float)

    if b.shape != (120, 3):
        raise ValueError(f"beta shape is {b.shape}, expected (120, 3)")
    return b


def get_mixture(rgb1, rgb2, t):
    """
    Take inputs rgb1, rgb2 as RGB triplets [0,255] and weighting t in [0,1]
    (weight for second color), and compute mixture RGB based on cubic fit.

    Returns uint8-like ints in [0,255] as a length-3 numpy array.
    """
    rgb1 = np.asarray(rgb1, dtype=float).reshape(-1)
    rgb2 = np.asarray(rgb2, dtype=float).reshape(-1)
    if rgb1.size != 3 or rgb2.size != 3:
        raise ValueError("rgb1 and rgb2 must be length-3.")

    t = float(t)
    t = np.clip(t, 0, 1).astype(float)

    # Fits were computed on [0,1]
    X = np.concatenate([rgb1 / 255.0, rgb2 / 255.0, [t]])  # shape (7,)

    # Map to [-1,1] and build features (1 x 120)
    Phi = poly_features_td3(2.0 * X - 1.0)  # shape (1,120)

    beta = load_beta()  # (120,3)

    # Vectorized: (1,120) @ (120,3) = (1,3)
    rgb = (Phi @ beta).reshape(3)

    # Back to [0,255] integer RGB
    rgb = np.rint(rgb * 255.0)
    rgb = np.clip(rgb, 0, 255).astype(int)

    return rgb


def generate_test_data(n_samples=1000):
    """Generate test samples across various mixing ratios."""
    print(f"Generating {n_samples} test samples (Ground Truth: Mixbox)...")
    X = []
    y = []
    t_values = []
    
    for _ in range(n_samples):
        c1 = list(np.random.randint(0, 256, 3))
        c2 = list(np.random.randint(0, 256, 3))
        t = np.random.random()
        
        # Ground truth from mixbox
        mixed = mixbox.lerp(tuple(c1), tuple(c2), t)
        
        X.append([c1, c2, t])
        y.append(mixed)
        t_values.append(t)
        
    return X, np.array(y), np.array(t_values)


def evaluate_wondermixer(n_test=1000):
    """Evaluate the wondermixer model against mixbox ground truth."""
    X_test, y_test, t_values = generate_test_data(n_test)
    
    print(f"\nEvaluating Wondermixer on {n_test} test samples...")
    
    start_time = time.time()
    predictions = []
    for c1, c2, t in X_test:
        pred = get_mixture(c1, c2, t)
        predictions.append(pred)
    predictions = np.array(predictions)
    duration = time.time() - start_time
    
    # Calculate Delta-E
    total_dE = 0
    dE_by_range = {'low': [], 'mid': [], 'high': []}  # Track performance by mixing ratio
    
    for i in range(n_test):
        rgb_true = y_test[i].reshape(1, 1, 3) / 255.0
        rgb_pred = predictions[i].reshape(1, 1, 3) / 255.0
        lab_true = color.rgb2lab(rgb_true)
        lab_pred = color.rgb2lab(rgb_pred)
        dE = np.sqrt(np.sum((lab_true - lab_pred)**2))
        total_dE += dE
        
        # Categorize by mixing ratio
        t = t_values[i]
        if t < 0.2 or t > 0.8:
            dE_by_range['low'].append(dE)
        elif 0.35 <= t <= 0.65:
            dE_by_range['mid'].append(dE)
        else:
            dE_by_range['high'].append(dE)
    
    mean_dE = total_dE / n_test
    
    print(f"\nResults (Exp E - Wondermixer):")
    print(f"  Mean Delta-E: {mean_dE:.4f}")
    print(f"  Inference Time: {duration*1000/n_test:.4f} ms/sample")
    print(f"\nDelta-E by mixing ratio:")
    print(f"  Extreme ratios (<0.2 or >0.8): {np.mean(dE_by_range['low']):.4f}")
    print(f"  Mid-range (0.35-0.65): {np.mean(dE_by_range['mid']):.4f}")
    print(f"  Other ratios: {np.mean(dE_by_range['high']):.4f}")
    
    return mean_dE


def test_edge_cases():
    """Test specific edge cases mentioned in the discussion."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    # Test: 100% red, 0% white should not be pink
    print("\nEdge Case 1: 100% red + 0% white (t=0)")
    red = [255, 0, 0]
    white = [255, 255, 255]
    result = get_mixture(red, white, 0.0)
    mixbox_result = mixbox.lerp(tuple(red), tuple(white), 0.0)
    print(f"  Wondermixer: {result}")
    print(f"  Mixbox:      {list(mixbox_result)}")
    print(f"  Expected:    [255, 0, 0]")
    
    # Test: 0% red, 100% white
    print("\nEdge Case 2: 0% red + 100% white (t=1)")
    result = get_mixture(red, white, 1.0)
    mixbox_result = mixbox.lerp(tuple(red), tuple(white), 1.0)
    print(f"  Wondermixer: {result}")
    print(f"  Mixbox:      {list(mixbox_result)}")
    print(f"  Expected:    [255, 255, 255]")
    
    # Test: Yellow + Blue at 0.5 (from original example)
    print("\nExample: Yellow + Blue (t=0.5)")
    yellow = [255, 255, 0]
    blue = [0, 0, 255]
    result = get_mixture(yellow, blue, 0.5)
    mixbox_result = mixbox.lerp(tuple(yellow), tuple(blue), 0.5)
    print(f"  Wondermixer: {result}")
    print(f"  Mixbox:      {list(mixbox_result)}")
    
    # Test several extreme ratios
    print("\nExtreme Ratios Test (Red + White):")
    for t in [0.05, 0.1, 0.9, 0.95]:
        result = get_mixture(red, white, t)
        mixbox_result = mixbox.lerp(tuple(red), tuple(white), t)
        dE = calculate_delta_e(result, mixbox_result)
        print(f"  t={t:.2f}: Wonder={result}, Mixbox={list(mixbox_result)}, ΔE={dE:.2f}")


def calculate_delta_e(rgb1, rgb2):
    """Calculate Delta-E between two RGB colors."""
    rgb1_norm = np.array(rgb1).reshape(1, 1, 3) / 255.0
    rgb2_norm = np.array(rgb2).reshape(1, 1, 3) / 255.0
    lab1 = color.rgb2lab(rgb1_norm)
    lab2 = color.rgb2lab(rgb2_norm)
    return np.sqrt(np.sum((lab1 - lab2)**2))


def main():
    print("="*60)
    print("Experiment E: Wondermixer Evaluation")
    print("="*60)
    print("\nA pre-fitted cubic polynomial model for color mixing")
    print("Approximate fit to MIXBOX (evaluated against MIXBOX ground truth)")
    print("Only requires: numpy")
    print("="*60)
    
    # Run evaluation
    evaluate_wondermixer(n_test=5000)
    
    # Test edge cases
    test_edge_cases()
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
