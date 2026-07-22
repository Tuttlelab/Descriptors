"""
rdp.py: Shared radial density profiling and hollowness utilities for descriptors.
"""
import numpy as np

def compute_radial_density(positions, com=None, num_bins=50):
    if com is None:
        com = np.mean(positions, axis=0)
    r = np.linalg.norm(positions - com, axis=1)
    density, bin_edges = np.histogram(r, bins=num_bins, density=True)
    return density, bin_edges

def is_hollow(density, bin_edges=None, window_size=7, hollow_ratio=0.05):
    if len(density) < window_size:
        return False
    kernel = np.ones(window_size)/window_size
    density_smooth = np.convolve(density, kernel, mode='same')
    center_bin = len(density_smooth) // 2
    window = density_smooth[center_bin-window_size//2:center_bin+window_size//2+1]
    if len(window) == 0:
        return False
    min_density = np.min(window)
    max_density = np.max(density_smooth)
    return min_density < hollow_ratio * max_density
