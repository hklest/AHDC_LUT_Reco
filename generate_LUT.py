#!/usr/bin/env python3
"""
AHDC_LUT_generation.py

Generates a lookup table (LUT) for AHDC track reconstruction.
The LUT maps track parameters (momentum, polar angle, and phi_phase) 
to the expected hit pattern (bit-packed, one bit per wire).
A phi_phase parameter is used instead of a full azimuth scan (since 
the chamber is nearly rotationally symmetric) so that only the relative 
phase between the track and the wires is stored.

The LUT is saved in a compressed numpy file (AHDC_LUT.npz) along with 
the bin edges.
"""

import numpy as np
import math
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Global Geometry and Simulation Parameters ---
# Fixed chamber geometry (mm)
z_geom_near = 0.0        # near endplate
z_geom_far = 300.0       # far endplate (300 mm)

# Track parameters (z vertex is provided externally)
track_z0 = 50.0          # fixed z vertex (mm)
B_field = 5.0            # Tesla

# Hit threshold (mm)
hit_threshold = 2.4

# AHDC wire mapping: superlayer -> (n_layers, numWires, base_radius)
layer_mapping = {
    1: {"n_layers": 1, "numWires": 47, "base_radius": 32.0},
    2: {"n_layers": 2, "numWires": 56, "base_radius": 38.0},
    3: {"n_layers": 2, "numWires": 72, "base_radius": 48.0},
    4: {"n_layers": 2, "numWires": 87, "base_radius": 58.0},
    5: {"n_layers": 1, "numWires": 99, "base_radius": 68.0},
}
DR_layer = 4.0  # mm increment per additional layer

# --- LUT Grid parameters ---
# (For demonstration we use smaller grids; in production you can use 500 bins for p and theta.)
N_P = 50         # number of momentum bins
N_THETA = 50     # number of polar angle bins
N_PHI = 10       # number of phi_phase bins (phi_phase in [0, 10) degrees now)

# Define ranges:
p_min, p_max = 0.1, 1.0          # momentum in GeV/c
theta_min, theta_max = 30.0, 60.0  # polar angle in degrees
phi_phase_min, phi_phase_max = 0.0, 10.0  # phi_phase in degrees (reduced range)

p_bins = np.linspace(p_min, p_max, N_P)
theta_bins = np.linspace(theta_min, theta_max, N_THETA)
phi_phase_bins = np.linspace(phi_phase_min, phi_phase_max, N_PHI, endpoint=False)

# Total number of wires (we define the ordering later)
def total_wires():
    tot = 0
    for sl in sorted(layer_mapping.keys()):
        mapping = layer_mapping[sl]
        tot += mapping["n_layers"] * mapping["numWires"]
    return tot
N_wires = total_wires()
# We pack bits: each LUT entry is an array of uint8 of length n_bytes = ceil(N_wires/8)
n_bytes = (N_wires + 7) // 8

# --- Utility Functions ---
def rotate_point(x, y, angle_deg):
    """Rotate point (x,y) by angle (in degrees) counterclockwise."""
    theta = math.radians(angle_deg)
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return xr, yr

def wire_position(superLayer, localLayer, wire):
    """
    Compute the nominal (x,y) position on the near endplate (z = 0) 
    for a given wire.
    """
    if superLayer not in layer_mapping:
        return 9999, 9999
    mapping = layer_mapping[superLayer]
    numWires = mapping["numWires"]
    R_layer = mapping["base_radius"] + DR_layer * (localLayer - 1)
    alphaW = (2.0 * math.pi) / numWires
    angle = alphaW * (wire - 1) + 0.5 * math.radians(20.0) * ((-1) ** (superLayer - 1))
    x = -R_layer * math.sin(angle)
    y = -R_layer * math.cos(angle)
    return x, y

def stereo_position(nominal_xy, stereo_offset_deg):
    """
    Rotate the nominal (x,y) position by the given stereo offset (in degrees).
    """
    theta = math.radians(stereo_offset_deg)
    x0, y0 = nominal_xy
    x = x0 * math.cos(theta) - y0 * math.sin(theta)
    y = x0 * math.sin(theta) + y0 * math.cos(theta)
    return x, y

def get_stereo_offset(superLayer, localLayer):
    """
    Returns the stereo offset for a given superLayer and localLayer.
    For superlayers 1 and 5, returns +10.
    For superlayers 2-4, returns +10 for localLayer 1 and -10 for localLayer 2.
    """
    stereo_sets = {1: [10], 2: [10, -10], 3: [10, -10], 4: [10, -10], 5: [10]}
    if superLayer in stereo_sets:
        offsets = stereo_sets[superLayer]
        if len(offsets) == 1:
            return offsets[0]
        elif len(offsets) >= 2:
            return offsets[0] if localLayer == 1 else offsets[1]
    return 0.0

# --- Track simulation (analytical distance calculation) ---
def point_to_segment_distance(P, A, B):
    """
    Compute the distance from point P to the finite segment AB.
    """
    AB = B - A
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    if t < 0:
        closest = A
    elif t > 1:
        closest = B
    else:
        closest = A + t * AB
    return np.linalg.norm(P - closest)

def helix_point(t, theta, z0, p, B_field, phi=0.0):
    """
    Return the (x,y,z) point on the helix at path length t.
    We assume track phi=0 (since phi_phase is applied to the wires).
    For B_field > 0.
    """
    theta_rad = theta
    pT = p * math.sin(theta_rad)
    R_val = (pT / (0.3 * B_field)) * 1000.0  # mm
    center_x = R_val * math.sin(0.0)
    center_y = -R_val * math.cos(0.0)
    x = center_x - R_val * math.sin(0.0 - t / R_val)
    y = center_y + R_val * math.cos(0.0 - t / R_val)
    z = z0 + t * math.cos(theta_rad)
    return np.array([x, y, z])

def compute_min_distance(P1, P2, p, theta, z0):
    """
    Compute minimum distance between wire segment [P1, P2] and the track.
    We simulate the track using a grid in path-length t.
    (Assumes B_field > 0 and track phi=0.)
    """
    theta_rad = theta
    t_max = (z_geom_far - z0) / math.cos(theta_rad)
    ts = np.linspace(0, t_max, 200)
    distances = []
    for t in ts:
        H = helix_point(t, theta_rad, z0, p, B_field, phi=0.0)
        distances.append(point_to_segment_distance(H, P1, P2))
    return min(distances)

# --- Function to simulate the hit pattern for one set of parameters ---
def simulate_hit_pattern(p, theta, phi_phase):
    """
    Simulate the hit pattern (as a boolean vector of length N_wires) for a track with
    momentum p (GeV/c) and polar angle theta (in degrees) using track phi = 0.
    The wire geometry is rotated by phi_phase (in degrees).
    """
    theta_rad = math.radians(theta)
    hit_pattern = []
    for sl in sorted(layer_mapping.keys()):
        mapping = layer_mapping[sl]
        n_layers = mapping["n_layers"]
        numWires = mapping["numWires"]
        for la in range(1, n_layers+1):
            stereo_off = get_stereo_offset(sl, la)
            for w in range(1, numWires+1):
                nom_xy = wire_position(sl, la, w)
                near_x, near_y = rotate_point(nom_xy[0], nom_xy[1], phi_phase)
                P1 = np.array([near_x, near_y, z_geom_near])
                far_nom_xy = stereo_position(nom_xy, stereo_off)
                far_x, far_y = rotate_point(far_nom_xy[0], far_nom_xy[1], phi_phase)
                P2 = np.array([far_x, far_y, z_geom_far])
                d = compute_min_distance(P1, P2, p, theta_rad, track_z0)
                hit_pattern.append(d < hit_threshold)
    return np.array(hit_pattern, dtype=bool)

def pack_hit_pattern(hit_pattern):
    """
    Pack a boolean hit pattern (length N_wires) into a uint8 array of length n_bytes.
    """
    return np.packbits(hit_pattern)

def compute_lut_entry(params):
    p_val, theta_val, phi_phase_val = params
    hit_bool = simulate_hit_pattern(p_val, theta_val, phi_phase_val)
    return pack_hit_pattern(hit_bool)

def main():
    # Create list of all parameter combinations.
    tasks = list(itertools.product(p_bins, theta_bins, phi_phase_bins))
    total_tasks = len(tasks)
    print("Generating LUT with parallel processing ...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_lut_entry, tasks),
                            total=total_tasks, desc="LUT entries"))
    LUT = np.empty((N_P, N_THETA, N_PHI, n_bytes), dtype=np.uint8)
    for index, entry in enumerate(results):
        i = index // (N_THETA * N_PHI)
        j = (index % (N_THETA * N_PHI)) // N_PHI
        k = index % N_PHI
        LUT[i, j, k, :] = entry
    np.savez_compressed("AHDC_LUT.npz",
                        LUT=LUT,
                        p_bins=p_bins,
                        theta_bins=theta_bins,
                        phi_phase_bins=phi_phase_bins)
    print("LUT generation complete. Saved to AHDC_LUT.npz")

if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError as e:
        print("Could not set start method to fork:", e)
    from multiprocessing import freeze_support
    freeze_support()
    main()
