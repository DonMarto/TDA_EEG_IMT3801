## Import necessary libraries
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

import mne
import numpy as np
import matplotlib.pyplot as plt
# import torch
from scipy.io import loadmat
from scipy.special import sph_harm
from numpy.linalg import solve
import json
from scipy.interpolate import RegularGridInterpolator

import numpy as np, matplotlib.pyplot as plt, os
from matplotlib import gridspec
from scipy import ndimage
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd
from sklearn.datasets import load_digits
try:
    from gtda.time_series import TakensEmbedding, SlidingWindow
    from gtda.homology import VietorisRipsPersistence
    from gtda.plotting import plot_diagram
    HAVE_GIOTTO = True
except Exception:
    HAVE_GIOTTO = False

good = [2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 33, 34, 36, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 65]
good = np.array(good) - 1
subjects = ["01", "02", "04", "09", "15", "37", "79"]
auds = [f"{i:02d}" for i in range(1, 24)]

bad = set(range(65)) - set(good)

san_disk = 'D:/Universidad/2025_2/TDA/data'
def takens_numpy(x, m=3, tau=10):
    """Takens embedding simple para una serie 1D -> matriz (N-(m-1)tau, m)."""
    N = len(x) - (m-1)*tau
    if N <= 0:
        raise ValueError("Serie muy corta para estos parámetros (m, tau).")
    return np.vstack([x[i:i+N] for i in range(0, m*tau, tau)]).T

n_phi = 128
n_theta = 64 

# Angular grids (depends on your definition!)
phi = np.linspace(-np.pi, np.pi, n_phi)
theta = np.linspace(0, np.pi, n_theta)

# Target Cartesian grid
N = 50
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Mask for the unit disk
mask = R <= 1

# Map to spherical angles (example: simple azimuthal projection)
phi_map = np.arctan2(Y, X)
theta_map = R * (np.pi / 2)


def process_case(args):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            m, n, speed, band = args
            bb = subjects[m]
            ut = auds[n]
            try:
                interp_maps = np.load(f"{san_disk}/eegs/harmonics/{speed}/{band}/bb{bb}_ut{ut}.npy")
                # print(f"Working on bb{bb}_ut{ut}, band {band}, speed {speed}")

            except FileNotFoundError:
                # print(f"File for bb{bb}_ut{ut} not found, skipping.")
                return None
            interp_maps = interp_maps[::10, :, :]
            movie = []
            for t in range(interp_maps.shape[0]):
                interpolator = RegularGridInterpolator((phi, theta), interp_maps[t], bounds_error=False, fill_value=0)
                pts = np.stack([phi_map[mask], theta_map[mask]], axis=-1)
                vals = interpolator(pts)
                frame = np.zeros((N, N))
                frame[mask] = vals
                movie.append(frame)
            movie = np.array(movie)
            
            
            for eid, emb in enumerate([3, 5, 7]):
                for tid, tau in enumerate(["", "_tau10"]):                

                    results_emb2 = np.zeros((interp_maps.shape[0] - (tid*3 + 1)*(emb-1), emb, 7, 7))
                    results_emb3 = np.zeros((interp_maps.shape[0] - (tid*3 + 1)*(emb-1), emb, 16, 8))


                    for i in range(7):
                        for j in range(7):
                            reel = movie[:, i*7, j*7]  
                            emb2 = takens_numpy(reel, m=emb, tau=(tid*3 + 1))
                            results_emb2[:, :, i, j] = emb2    

                    for i in range(16):
                        for j in range(8):
                            reel = interp_maps[:, i*8, j*8]  
                            emb3 = takens_numpy(reel, m=emb, tau=(tid*3 + 1))
                            results_emb3[:, :, i, j] = emb3

                    with open(f'{san_disk}/eegs/method_2/{band}/{speed}/bb{bb}_ut{ut}_emb{emb}{tau}_square.json', 'w') as f:
                        json.dump(results_emb2.tolist(), f)

                    with open(f'{san_disk}/eegs/method_2/{band}/{speed}/bb{bb}_ut{ut}_emb{emb}{tau}_sphere.json', 'w') as f:
                        json.dump(results_emb3.tolist(), f)
            return bb, ut, band, speed

        except Exception as e:
                print(f"❌ Error in task {args}: {e}")
                return None
        

if __name__ == "__main__":
    # san_disk = "/your/path"
    tasks = []
    print("Starting")

    tasks = [(m, n, speed, band) 
             for m in range(1,len(subjects))
             for n in range(len(auds))
             for speed in ["slow", "fast"]
             for band in ["theta", "alpha", "beta"]]

    print("Tasks Assigned")

    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_case, tasks), total=len(tasks)):
            pass

    print("✔️ Finished all cases!")
