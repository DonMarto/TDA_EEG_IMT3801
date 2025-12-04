import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np
from numpy.linalg import solve

import os
import mne
import librosa
import json
import pandas as pd

from scipy.signal import hilbert, butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.special import sph_harm
from scipy.signal import resample
from scipy.stats import pearsonr
from scipy import ndimage
## Import necessary libraries

import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.manifold import MDS

from ripser import ripser
from persim import plot_diagrams
from persim import wasserstein, bottleneck
import gudhi as gd

try:
    from gtda.time_series import TakensEmbedding, SlidingWindow
    from gtda.homology import VietorisRipsPersistence
    from gtda.plotting import plot_diagram
    from gtda.diagrams import PersistenceLandscape, PersistenceSilhouette
    from gtda.diagrams import Scaler, Filtering
    HAVE_GIOTTO = True
except Exception:
    HAVE_GIOTTO = False

good = [2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 33, 34, 36, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 65]
good = np.array(good) - 1
subjects = ["01", "02", "04", "09", "15", "37", "79"]
auds = [f"{i:02d}" for i in range(1, 24)]

bad = set(range(65)) - set(good)

san_disk = 'D:/Universidad/2025_2/TDA/data'
san_results = 'D:/Universidad/2025_2/TDA/results'

embeds = [30, 50, 64]
taus = ["", "_tau10"]
bands = ["alpha", "theta", "beta"]
typps = ["sphere", "square"]

# df_scores = pd.DataFrame(columns=["Model", "Method", "Subject", "Trial", "Speed", "Band", "Emb", "Tau", "Type", "Score"])
# df_scores.to_csv("scores_4.csv")

# df_scores = pd.DataFrame(columns=["Model", "Method", "Subject", "Trial", "Speed", "Band", "Emb", "Tau", "Type", "Score"])
# df_scores.to_csv("scores_5.csv")

# df_scores = pd.DataFrame(columns=["Model", "Method", "Subject", "Trial", "Speed", "Band", "Emb", "Tau", "Type", "Score"])
# df_scores.to_csv("scores_6.csv")

prop = np.zeros((7,23))



for m in range(7):
    bb = subjects[m]
    for n in range(len(auds)):
        ut = auds[n]
        considering = 1
        for s in range(2):
            if s == 0:
                speed = 'fast'
            else:
                speed = 'slow'
            try:
                data = loadmat(f'data/sound_sep/{speed}/bb{bb}_ut{ut}.mat')
            except FileNotFoundError:
                considering = 0
                continue
        if considering:
            prop[m,n] = 1


# ---------------------------------------------------------
# 1. Hilbert envelope
# ---------------------------------------------------------
def envelope_hilbert(y, sr, smooth_ms=10):
    """Hilbert transform envelope, optionally smoothed with moving average."""
    analytic = hilbert(y)
    env = np.abs(analytic)
    win = int(smooth_ms * sr / 1000)
    if win > 1:
        env = np.convolve(env, np.ones(win)/win, mode='same')
    return env


# ---------------------------------------------------------
# 2. Rectify + Lowpass filter envelope
# ---------------------------------------------------------
def envelope_lowpass(y, sr, cutoff_hz=20, order=4):
    """Envelope via full-wave rectification and Butterworth lowpass."""
    rect = np.abs(y)
    nyq = 0.5 * sr
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    env = filtfilt(b, a, rect)
    return env


# ---------------------------------------------------------
# 3. RMS (root-mean-square) envelope
# ---------------------------------------------------------
def envelope_rms(y, sr, frame_ms=128, hop_ms=None):
    """RMS energy envelope over frames."""
    frame = int(frame_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000) if hop_ms else frame // 2
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    # Upsample to original signal length
    env = np.repeat(rms, hop)
    return env[:len(y)]


# ---------------------------------------------------------
# 4. Peak interpolation envelope
# ---------------------------------------------------------
def envelope_peaks(y, sr, peak_thresh=0.01, min_dist_ms=20):
    """Envelope by finding peaks and interpolating between them."""
    rect = np.abs(y)
    min_dist = int(min_dist_ms * sr / 1000)
    peaks, _ = find_peaks(rect, height=np.max(rect)*peak_thresh, distance=min_dist)

    if len(peaks) < 2:  # fallback: return rectified signal
        return rect

    xs = np.concatenate(([0], peaks, [len(y)-1]))
    ys = np.concatenate(([rect[0]], rect[peaks], [rect[-1]]))
    f = interp1d(xs, ys, kind='linear')
    env = f(np.arange(len(y)))
    return env


# ---------------------------------------------------------
# 5. Exponential smoothing envelope (attack/release)
# ---------------------------------------------------------
def envelope_exponential(y, sr, attack_ms=1, release_ms=200):
    """Envelope follower with exponential attack/release smoothing."""
    x = np.abs(y)
    a_a = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    a_r = np.exp(-1.0 / (sr * release_ms / 1000.0))
    env = np.zeros_like(x)
    for n in range(1, len(x)):
        coeff = a_a if x[n] > env[n-1] else a_r
        env[n] = coeff * env[n-1] + (1 - coeff) * x[n]
    return env


# D matriz T x T con distancias Wasserstein
def pairwise_wasserstein(pd_list, matching=False):
    T = len(pd_list)
    D = np.zeros((T,T))
    for i in range(T):
        for j in range(i+1, T):
            d = wasserstein(pd_list[i], pd_list[j], matching=matching)
            D[i,j] = d
            D[j,i] = d
    return D
def takens_numpy(x, m=3, tau=10):
    """Takens embedding simple para una serie 1D -> matriz (N-(m-1)tau, m)."""
    N = len(x) - (m-1)*tau
    if N <= 0:
        raise ValueError("Serie muy corta para estos parámetros (m, tau).")
    return np.vstack([x[i:i+N] for i in range(0, m*tau, tau)]).T


def takens_wasser(pd_list, m=3, tau=1, matching = False):
    T = len(pd_list) - (m-1)*tau
    if T <= 0:
        return
        # raise ValueError("Serie muy corta para estos parámetros (m, tau).")
    D = np.zeros((T,m))
    for i in range(0, T):
        for j in range(m):
            d = wasserstein(pd_list[i], pd_list[i + j*tau], matching=matching)
            D[i,j] = d
    return D

# embedding from D via MDS
def embed_from_distance_matrix(D, n_components=10):
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=0)
    Y = mds.fit_transform(D)
    return Y  # T x n_components

def combine_diagrams(Ds):
    """
    Convert [D0, D1] → (n_points_total, 3) array with homology dimension labels.
    """
    D0, D1 = Ds
    D0 = np.asarray(D0, dtype=float)
    D1 = np.asarray(D1, dtype=float)

    # Add homology dimension label as 3rd column
    D0_labeled = np.hstack([D0, np.zeros((len(D0), 1))])
    D1_labeled = np.hstack([D1, np.ones((len(D1), 1))])

    # Concatenate into one array
    return np.vstack([D0_labeled, D1_labeled])

def pd_basic_features(D):
    births, deaths = D[:,0], D[:,1]
    lifetimes = deaths - births
    return {
        "num": len(D),
        "mean": np.mean(lifetimes),
        "max": np.max(lifetimes),
        "sum": np.sum(lifetimes),
        "std": np.std(lifetimes)
    }

def persistence_entropy(D):
    lifetimes = D[:,1] - D[:,0]
    if np.sum(lifetimes) == 0:
        return 0.0
    p = lifetimes / np.sum(lifetimes)
    return -np.sum(p * np.log(p + 1e-10))

def betti_curve_from_diagram(diag: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Cuenta, para cada t en t_grid, cuántos intervalos (b,d) activan: b <= t < d.
    diag: array de pares (b,d) (puede contener d = inf)"""
    if len(diag) == 0:
        return np.zeros_like(t_grid, dtype=float)
    b = [case[0] for case in diag] # todos los births
    d = [case[1] for case in diag] # todos los deaths
    out = np.zeros_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        out[i] = np.sum((b <= t) & (t < d)) # número de intervalos de persistencia "vivos" en el tiempo t
    return out

def euler_curve_from_diagrams(diagrams: list[np.ndarray], t_grid: np.ndarray) -> np.ndarray:
    """E(t) = sum_k (-1)^k * beta_k(t)"""
    E = np.zeros_like(t_grid, dtype=float)
    for k, Dk in enumerate(diagrams):
        if len(Dk) == 0:
            continue
        E += ((-1)**k) * betti_curve_from_diagram(Dk, t_grid)
    return E


# ---------------------
# Paisajes de persistencia (versión simple)
# ---------------------
def _hat_height(t, b, d):
    # Triangular hat: max(0, min(t-b, d-t))
    if np.isinf(d):
        # Si la muerte es infinita, acotamos artificialmente (para visualización)
        d = b + 2.0
    return max(0.0, min(t - b, d - t))

def persistence_landscapes(diag: np.ndarray, t_grid: np.ndarray, k_max=3) -> np.ndarray:
    """Devuelve una matriz (k_max, len(t_grid)) con los k primeros paisajes."""
    L = np.zeros((k_max, len(t_grid)), dtype=float)
    if len(diag) == 0:
        return L
    for j, t in enumerate(t_grid):
        vals = [_hat_height(t, case[0], case[1]) for case in diag]
        vals.sort(reverse=True)
        m = min(k_max, len(vals))
        L[:m, j] = vals[:m]
    return L

def features_from_pd_list(D_list):
        feats = []
        t_grid = np.linspace(-1e-15, 2e-15, 50)
        for D in D_list:
            # min_t, max_t = D[:,0].min(), D[:,1].max()
            # t_grid = np.linspace(min_t, max_t, 50)
            b0 = betti_curve_from_diagram([case for case in D if case[2] < 0.5], t_grid) if len(D)>0 else np.zeros(t_grid)
            b1 = betti_curve_from_diagram([case for case in D if case[2] > 0.5], t_grid) if len(D)>1 else np.zeros(t_grid)
            E  = euler_curve_from_diagrams([[case for case in D if case[2] < 0.5], [case for case in D if case[2] > 0.5]], t_grid)
            L0  = persistence_landscapes([case for case in D if case[2] < 0.5], t_grid, k_max=3) if len(D)>1 else np.zeros((3,len(t_grid)))
            L1  = persistence_landscapes([case for case in D if case[2] > 0.5], t_grid, k_max=3) if len(D)>1 else np.zeros((3,len(t_grid)))
            lifetimes_0 = np.array([d - b for (b, d, dim) in D if dim == 0])
            finite_mask_0 = np.isfinite(lifetimes_0)
            filtered_arr_0 = lifetimes_0[finite_mask_0]

            mean_life_0 = filtered_arr_0.mean() if len(filtered_arr_0) > 0 else 0
            max_life_0 = filtered_arr_0.max() if len(filtered_arr_0) > 0 else 0
            var_life_0 = filtered_arr_0.var() if len(filtered_arr_0) > 0 else 0

            lifetimes_1 = np.array([d - b for (b, d, dim) in D if dim == 0])
            finite_mask_1 = np.isfinite(lifetimes_1)
            filtered_arr_1 = lifetimes_1[finite_mask_1]

            mean_life_1 = filtered_arr_1.mean() if len(filtered_arr_1) > 0 else 0
            max_life_1 = filtered_arr_1.max() if len(filtered_arr_1) > 0 else 0
            var_life_1 = filtered_arr_1.var() if len(filtered_arr_1) > 0 else 0

            num_infinite_0 = np.sum([np.isinf(d) for (_, d, dim) in D if dim == 0])
            num_infinite_1 = np.sum([np.isinf(d) for (_, d, dim) in D if dim == 1])

            if len(filtered_arr_0) > 0:
                probs_0 = filtered_arr_0 / filtered_arr_0.sum()
                entropy_0 = -np.sum(probs_0 * np.log(probs_0))
            else:
                entropy_0 = 0

            if len(filtered_arr_1) > 0:
                probs_1 = filtered_arr_1 / filtered_arr_1.sum()
                entropy_1 = -np.sum(probs_1 * np.log(probs_1))
            else:
                entropy_1 = 0

            feats.append([
                mean_life_0,
                max_life_0,
                var_life_0,
                num_infinite_0,
                entropy_0,
                np.trapz(b0, t_grid),          # área bajo Betti0
                b0.max(),                      # peak Betti0
                L0[0].max() if L0.size>0 else 0, # peak λ0
                np.trapz(np.abs(L0[0]), t_grid),  # p=1
                np.sqrt(np.trapz(L0[0]**2, t_grid)),  # p=2

                mean_life_1,
                max_life_1,
                var_life_1,
                num_infinite_1,
                entropy_1,
                np.trapz(b1, t_grid),          # área bajo Betti1
                b1.max(),                      # peak Betti1
                L1[0].max() if L1.size>0 else 0, # peak λ1
                np.trapz(np.abs(L1[0]), t_grid),  # p=1
                np.sqrt(np.trapz(L1[0]**2, t_grid)),  # p=2
                
                np.trapz(np.abs(E), t_grid),   # área Euler
            ])
        return np.array(feats)

def cubical_pd_from_image(img):
    cc = gd.CubicalComplex(dimensions=img.shape, top_dimensional_cells=img.flatten())
    cc.compute_persistence()
    D0 = np.array(cc.persistence_intervals_in_dimension(0))
    D1 = np.array(cc.persistence_intervals_in_dimension(1))
    return [D0.tolist(), D1.tolist()]

def gudhi_rips_diagrams(X, maxdim=1, max_edge_length=None):
    max_edge = np.inf if max_edge_length is None else max_edge_length
    rc = gd.RipsComplex(points=X, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=maxdim+1)
    st.compute_persistence() # [(dim, [birth, death])]
    dgms = []
    # como gudhi tiene los intervalos (dim, [birth, death]) lo homogenizamos con el formato de ripser
    # una lista de listas de intervalos
    for dim in range(maxdim+1):
        d = st.persistence_intervals_in_dimension(dim)
        # Gudhi usa 'inf' para muertes infinitas; lo dejamos así para compatibilidad con persim
        dgms.append(np.array(d, dtype=float))
    return dgms


def regress_models_2(X_raw, y_raw, subjects, method = "Ridge", candidates = [{"alpha": 1}, {"alpha": 2}], verb=0):
    
    target_len = X_raw[0].shape[0] # pick desired uniform length
    X_fixed = []
    y_fixed = []

    for X, y in zip(X_raw, y_raw):
        # Resample EEG (each channel)
        Xr = X[:target_len, :]  # truncate or pad as needed
        # Flatten: channels * time
        X_flat = Xr.reshape(-1)
        X_fixed.append(X_flat)

        # Resample audio
        yr = y[:target_len]
        y_flat = yr.reshape(-1)  # truncate or pad as needed
        y_fixed.append(y_flat)

    X_fixed = np.array(X_fixed)
    y_fixed = np.array(y_fixed)
    subjects = np.array(subjects)
    if verb:
        print(X_fixed.shape, y_fixed.shape, subjects.shape)


    #########################################
    # 3) Train/Test subject split
    #########################################

    train_subjects = [1, 2, 3, 4]  # example subjects for training
    test_subjects  = [0, 6]           # held-out test subjects

    train_mask = np.isin(subjects, train_subjects)
    test_mask  = np.isin(subjects, test_subjects)

    X_train, y_train = X_fixed[train_mask], y_fixed[train_mask]
    X_test,  y_test  = X_fixed[test_mask],  y_fixed[test_mask]
    groups_train    = subjects[train_mask]


    #########################################
    # 4) Standardization (fit only on training!)
    #########################################

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_train_scaled = X_train_scaled
    X_test_scaled = X_test_scaled
    y_train = y_train
    y_test = y_test


    #########################################
    # 5) GroupKFold CV on training subjects
    #########################################

    gkf = GroupKFold(n_splits=len(train_subjects))


    best_score = -np.inf
    best_params = 0
    if verb:
        print("Starting Train Stage")
    for n_id, n_comp in enumerate(candidates):
        fold_scores = []

        for train_idx, val_idx in gkf.split(X_train_scaled, y_train, groups_train):
            if method == "Ridge":
                model = Ridge(**n_comp)
            elif method == "PLS":
                model = PLSRegression(**n_comp)
            elif method == "KRidge":
                model = KernelRidge(**n_comp)
            elif method == "MLP":
                model = MLPRegressor(**n_comp)
            model.fit(X_train_scaled[train_idx], y_train[train_idx])
            y_pred = model.predict(X_train_scaled[val_idx])

            corrs = [pearsonr(y_pred[i].ravel(), y_train[val_idx][i].ravel())[0]
                    for i in range(len(val_idx))]
            fold_scores.append(np.nanmean(corrs))

        avg_score = np.mean(fold_scores)


        if avg_score > best_score:
            best_score = avg_score
            best_params = n_id
    if verb:
        print("\nBest hyperparameters:", best_params)
        print("Best CV Score:", best_score)


    #########################################
    # 6) Retrain final model on all training subjects
    #########################################
    if verb:
        print("Starting Test Stage")
    if method == "Ridge":
        final_model = Ridge(**candidates[best_params])
    elif method == "PLS":
        final_model = PLSRegression(**candidates[best_params])
    elif method == "KRidge":
        final_model = KernelRidge(**candidates[best_params])
    elif method == "MLP":
        final_model = MLPRegressor(**candidates[best_params])
    final_model.fit(X_train_scaled, y_train)

    #########################################
    # 7) Evaluate on held-out test subjects
    #########################################

    y_test_pred = final_model.predict(X_test_scaled)

    corrs = [pearsonr(y_test_pred[i].ravel(), y_test[i].ravel())[0]
                        for i in range(len(y_test_pred))]

    test_score = np.array(corrs).mean()

    if verb:
        print("\nFinal Test Score:", test_score)

    return y_test, y_test_pred, test_score, corrs


def process_case(args):
    print("hi")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            eid, tid, bid, tyd = args
            print(f"Working on case Embedding {eid}, Tau {tid}, Band {bid}, Type {tyd}")
            emb = embeds[eid]
            tau = taus[tid]
            band = bands[bid]
            typp = typps[tyd]





            XF = []
            yF1 = []
            yF2 = []
            XS = []
            yS1 = []
            yS2 = []
            gF = []
            gS = []

            for m in range(len(subjects)):
                bb = subjects[m]
                # print(f"Working on subject {bb}")
                for n in range(len(auds)):
                        
                    ut = auds[n]
                    # Load audio file
                    if prop[m,n]:
                        for s in range(2):
                            if s == 0:
                                speed = 'fast'
                            else:
                                speed = 'slow'

                    # for eid, emb in enumerate([30, 50, 64]):
                    #     for tid, tau in enumerate(["", "_tau10"]):
                            with open(f'{san_disk}/audios/method_1/{speed}/ut{ut}_emb{emb}{tau}_hilbert.json', 'r') as file:
                                aud = json.load(file)
                            aud = np.array(aud)
                                    
                            eeg = []
                            # for bid, band in enumerate(["alpha", "beta", "theta"]):
                            
                            if typp == "sphere":
                                with open(f'{san_disk}/eegs/method_3/{band}/{speed}/bb{bb}_ut{ut}_cc_{typp}.json', 'r') as file:
                                    eeg_data = json.load(file)
                            
                                eeg = []
                                for D_idx in range(0,len(eeg_data),int(250/64)):
                                    Ds = eeg_data[D_idx]
                                    eeg.append(combine_diagrams(Ds))  
                                eeg = features_from_pd_list(eeg)
                            
                            else:
                                eeg = np.load(f'{san_disk}/eegs/method_4_1/{band}/{speed}/bb{bb}_ut{ut}_cc_{typp}.npy')

                                #
                            aud2 = ripser(aud, maxdim=1)  # compute up to H2
                            aud2 = aud2['dgms']
                            aud2 = combine_diagrams(aud2)
                            aud2 = features_from_pd_list([aud2])
                            # aud = aud[::3,:]
                            # print(aud.shape)
                            # print(eeg.shape)
                            

                            # print(eeg.shape)
                            # print(aud.shape)
                            # eeg = eeg[::int(250/64)]
                            # aud = aud[::10]
                            if eeg.shape[0] > aud.shape[0]:
                                eeg = eeg[:aud.shape[0]]
                            elif eeg.shape[0] < aud.shape[0]:
                                aud = aud[:eeg.shape[0]]

                            # eeg = eeg.flatten()
                            # aud = aud.flatten()

                            # eeg = eeg[::59]
                            # aud = aud[::59]

                            if s == 0:
                                XF.append(eeg)
                                yF1.append(aud)
                                yF2.append(aud2)
                                gF.append(m)
                            else:
                                XS.append(eeg)
                                yS1.append(aud)
                                yS2.append(aud2)
                                gS.append(m)
            minim = 10000
            minimy = 10000

            for i in XF:
                if i.shape[0] < minim:
                    minim = i.shape[0]
                    # print(f"Current minimum {i.shape}")

            for i in yF1:
                if i.shape[0] < minimy:
                    minimy = i.shape[0]
                    # print(f"Current minimum {i.shape}")

            for i in range(len(XF)):
                XF[i] = XF[i][:minim]
                XS[i] = XS[i][:minim]
                yF1[i] = yF1[i][:minimy]
                yS1[i] = yS1[i][:minimy]

            XF = np.array(XF)
            XS = np.array(XS)
            yF1 = np.array(yF1)
            yF2 = np.array(yF2)
            yS1 = np.array(yS1)
            yS2 = np.array(yS2)
            gF = np.array(gF)
            gS = np.array(gS)

            XF2 = []
            for i in XF:
                lines = []
                for j in range(i.shape[1]):
                    line = takens_numpy(i[:,j])
                    lines.append(line)
                XF2.append(lines)
            XF2 = np.array(XF2)

            XS2 = []
            for i in XS:
                lines = []
                for j in range(i.shape[1]):
                    line = takens_numpy(i[:,j])
                    lines.append(line)
                XS2.append(lines)
            XS2 = np.array(XS2)            

            params = [{"alpha":2**i} for i in range(-5,5)]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF1, gF, method = "Ridge", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS1, gS, method = "Ridge", candidates=params)

            # if scoreF < scoreS:
            #     print(f"Slow outperforms Fast with Ridge")
            # else:
            #     print(f"Fast outperforms Slow with Ridge")

            rows = []

            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 4,
                            "Method": 0,
                            "Subject": m*6 + 1,
                            "Trial": n + 1,
                            "Speed": s,
                            "Band": bid,
                            "Emb": eid,
                            "Tau": tid,
                            "Type": tyd,
                            "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
                        }
                        rows.append(row)


            params = [{"alpha":2**i, "gamma": j} for i in range(-5,5) for j in [1e-3, 1e-2,5e-3]]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF1, gF, method = "KRidge", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS1, gS, method = "KRidge", candidates=params)

            # if scoreF < scoreS:
            #     print(f"Slow outperforms Fast with KernelRidge")
            # else:
            #     print(f"Fast outperforms Slow with KernelRidge")

            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 4,
                            "Method": 1,
                            "Subject": m*6 + 1,
                            "Trial": n + 1,
                            "Speed": s,
                            "Band": bid,
                            "Emb": eid,
                            "Tau": tid,
                            "Type": tyd,
                            "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
                        }
                        rows.append(row)
            
            
            params = [{"n_components":5*i} for i in range(1,12,3)]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF1, gF, method = "PLS", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS1, gS, method = "PLS", candidates=params)
            
            # if scoreF < scoreS:
            #     print(f"Slow outperforms Fast with PLS")
            # else:
            #     print(f"Fast outperforms Slow with PLS")
            
            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 4,
                            "Method": 2,
                            "Subject": m*6 + 1,
                            "Trial": n + 1,
                            "Speed": s,
                            "Band": bid,
                            "Emb": eid,
                            "Tau": tid,
                            "Type": tyd,
                            "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
                        }
                        rows.append(row)

            params = [{"hidden_layer_sizes":3*i, "max_iter": 600} for i in range(1,25,5)]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF1, gF, method = "MLP", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS1, gS, method = "MLP", candidates=params)

            # if scoreF < scoreS:
            #     print(f"Slow outperforms Fast with MLP")
            # else:
            #     print(f"Fast outperforms Slow with MLP")
            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 4,
                            "Method": 3,
                            "Subject": m*6 + 1,
                            "Trial": n + 1,
                            "Speed": s,
                            "Band": bid,
                            "Emb": eid,
                            "Tau": tid,
                            "Type": tyd,
                            "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
                        }
                        rows.append(row)


            df_scores = pd.read_csv("scores_4.csv")
            df_scores = pd.concat([df_scores, pd.DataFrame(rows)], ignore_index=True)
            df_scores.to_csv("scores_4.csv", index=False)
            

            # params = [{"alpha":2**i} for i in range(-5,5)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF2, gF, method = "Ridge", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS2, gS, method = "Ridge", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with Ridge")
            # # else:
            # #     print(f"Fast outperforms Slow with Ridge")

            # rows = []

            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 5,
            #                 "Method": 0,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)


            # params = [{"alpha":2**i, "gamma": j} for i in range(-5,5) for j in [1e-3, 1e-2,5e-3]]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF2, gF, method = "KRidge", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS2, gS, method = "KRidge", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with KernelRidge")
            # # else:
            # #     print(f"Fast outperforms Slow with KernelRidge")

            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 5,
            #                 "Method": 1,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)
            
            
            # params = [{"n_components":5*i} for i in range(1,12,3)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF2, gF, method = "PLS", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS2, gS, method = "PLS", candidates=params)
            
            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with PLS")
            # # else:
            # #     print(f"Fast outperforms Slow with PLS")
            
            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 5,
            #                 "Method": 2,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)

            # params = [{"hidden_layer_sizes":3*i, "max_iter": 600} for i in range(1,25,5)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF2, gF, method = "MLP", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS2, gS, method = "MLP", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with MLP")
            # # else:
            # #     print(f"Fast outperforms Slow with MLP")
            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 5,
            #                 "Method": 3,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)


            # df_scores = pd.read_csv("scores_5.csv")
            # df_scores = pd.concat([df_scores, pd.DataFrame(rows)], ignore_index=True)
            # df_scores.to_csv("scores_5.csv", index=False)


            # params = [{"alpha":2**i} for i in range(-5,5)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF2, yF1, gF, method = "Ridge", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS2, yS1, gS, method = "Ridge", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with Ridge")
            # # else:
            # #     print(f"Fast outperforms Slow with Ridge")

            # rows = []

            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 6,
            #                 "Method": 0,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)


            # params = [{"alpha":2**i, "gamma": j} for i in range(-5,5) for j in [1e-3, 1e-2,5e-3]]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF2, yF1, gF, method = "KRidge", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS2, yS1, gS, method = "KRidge", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with KernelRidge")
            # # else:
            # #     print(f"Fast outperforms Slow with KernelRidge")

            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 6,
            #                 "Method": 1,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)
            
            
            # params = [{"n_components":5*i} for i in range(1,12,3)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF2, yF1, gF, method = "PLS", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS2, yS1, gS, method = "PLS", candidates=params)
            
            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with PLS")
            # # else:
            # #     print(f"Fast outperforms Slow with PLS")
            
            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 6,
            #                 "Method": 2,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)

            # params = [{"hidden_layer_sizes":3*i, "max_iter": 600} for i in range(1,25,5)]

            # ytestF, ypredF, scoreF, pearF = regress_models_2(XF2, yF1, gF, method = "MLP", candidates=params)
            # ytestS, ypredS, scoreS, pearS = regress_models_2(XS2, yS1, gS, method = "MLP", candidates=params)

            # # if scoreF < scoreS:
            # #     print(f"Slow outperforms Fast with MLP")
            # # else:
            # #     print(f"Fast outperforms Slow with MLP")
            # for m in range(2):
            #     for n in range(23 - m):
            #         for s in range(2):
            #             row = {
            #                 "Model": 6,
            #                 "Method": 3,
            #                 "Subject": m*6 + 1,
            #                 "Trial": n + 1,
            #                 "Speed": s,
            #                 "Band": bid,
            #                 "Emb": eid,
            #                 "Tau": tid,
            #                 "Type": tyd,
            #                 "Score": pearF[23*m + n] if s == 0 else pearS[23*m + n]
            #             }
            #             rows.append(row)


            # df_scores = pd.read_csv("scores_6.csv")
            # df_scores = pd.concat([df_scores, pd.DataFrame(rows)], ignore_index=True)
            # df_scores.to_csv("scores_6.csv", index=False)
                

            return eid, tid, bid, tyd#, ut, band, speed

        except Exception as e:
                print(f"❌ Error in task {args}: {e}")
                return None
        

if __name__ == "__main__":
    # san_disk = "/your/path"
    tasks = []
    print("Starting")

    tasks = [(eid, tid, bid, tyd) 
             for eid in range(1)#([30, 50, 64]):
             for tid in range(1)# tau in enumerate(["", "_tau10"]):
             for bid in range(1)#, band in enumerate(["alpha", "theta", "beta"]):
             for tyd in range(1)]#, typp in enumerate(["sphere", "square"]):]

    print("Tasks Assigned")
    print(tasks)

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            pass

    print("✔️ Finished all cases!")
            
            
                
                
