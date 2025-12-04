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

from ripser import ripser
from persim import plot_diagrams
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

df_scores = pd.DataFrame(columns=["Model", "Method", "Subject", "Trial", "Speed", "Band", "Emb", "Tau", "Type", "Score"])
df_scores.to_csv("scores_2.csv")

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
            yF = []
            gF = []
            XS = []
            yS = []
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
                            with open(f'{san_disk}/eegs/method_2/{band}/{speed}/bb{bb}_ut{ut}_emb{int(emb/10 + emb%10/4)}{tau}_{typp}.json', 'r') as file:
                                eeg_data = json.load(file)
                            
                            eeg = np.array(eeg_data)
                            # print(eeg.shape)
                            # print(aud.shape)
                            eeg = eeg[::int(250/64)]
                            aud = aud[::10]
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
                                yF.append(aud)
                                gF.append(m)
                            else:
                                XS.append(eeg)
                                yS.append(aud)
                                gS.append(m)
            minim = 10000

            for i in range(153):#XF_aux:
                if yF[i].shape[0] < minim:
                    minim = yF[i].shape[0]
                    # print(f"Current mininum {yF[i].shape}")

            for i in range(len(XF)):
                XF[i] = XF[i][:minim]
                yF[i] = yF[i][:minim]
                XS[i] = XS[i][:minim]
                yS[i] = yS[i][:minim]
                gF[i] = gF[i]
                gS[i] = gS[i]

            XF = np.array(XF)
            yF = np.array(yF)
            gF = np.array(gF)
            XS = np.array(XS)
            yS = np.array(yS)
            gS = np.array(gS)

            params = [{"alpha":2**i} for i in range(-5,5)]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF, gF, method = "Ridge", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS, gS, method = "Ridge", candidates=params)

            if scoreF < scoreS:
                print(f"Slow outperforms Fast with Ridge")
            else:
                print(f"Fast outperforms Slow with Ridge")

            rows = []

            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 2,
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

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF, gF, method = "KRidge", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS, gS, method = "KRidge", candidates=params)

            if scoreF < scoreS:
                print(f"Slow outperforms Fast with KernelRidge")
            else:
                print(f"Fast outperforms Slow with KernelRidge")

            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 2,
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

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF, gF, method = "PLS", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS, gS, method = "PLS", candidates=params)
            
            if scoreF < scoreS:
                print(f"Slow outperforms Fast with PLS")
            else:
                print(f"Fast outperforms Slow with PLS")
            
            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 2,
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

            params = [{"hidden_layer_sizes":3*i} for i in range(1,25,5)]

            ytestF, ypredF, scoreF, pearF = regress_models_2(XF, yF, gF, method = "MLP", candidates=params)
            ytestS, ypredS, scoreS, pearS = regress_models_2(XS, yS, gS, method = "MLP", candidates=params)

            if scoreF < scoreS:
                print(f"Slow outperforms Fast with MLP")
            else:
                print(f"Fast outperforms Slow with MLP")
            for m in range(2):
                for n in range(23 - m):
                    for s in range(2):
                        row = {
                            "Model": 2,
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


            df_scores = pd.read_csv("scores_2.csv")
            df_scores = pd.concat([df_scores, pd.DataFrame(rows)], ignore_index=True)
            df_scores.to_csv("scores_2.csv", index=False)

            return eid, tid, bid, tyd#, ut, band, speed

        except Exception as e:
                print(f"❌ Error in task {args}: {e}")
                return None
        

if __name__ == "__main__":
    # san_disk = "/your/path"
    tasks = []
    print("Starting")

    tasks = [(eid, tid, bid, tyd) 
             for eid in range(3)#([30, 50, 64]):
             for tid in range(2)# tau in enumerate(["", "_tau10"]):
             for bid in range(3)#, band in enumerate(["alpha", "theta", "beta"]):
             for tyd in range(2)]#, typp in enumerate(["sphere", "square"]):]

    print("Tasks Assigned")

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            pass

    print("✔️ Finished all cases!")