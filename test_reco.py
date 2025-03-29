#!/usr/bin/env python3
"""
AHDC_LUT_tree_save.py

Simulates many tracks using the drift chamber simulation logic and your LUT‐based reconstruction.
For each event it saves:
  true_p, true_theta, true_phi, reco_p, reco_theta, reco_phi
into a ROOT TTree.
The TTree is saved to "AHDC_LUT_reco.root".
"""

import ROOT
from array import array
import random
import math
import numpy as np

# Import LUT reco functions (adjust module name/path as needed)
from LUT_reco import load_LUT, reconstruct_hit_pattern

# Global parameters (must match your LUT generation settings)
z_geom_near = 0.0
z_geom_far = 300.0
track_z0 = 50.0
B_field = 5.0
hit_threshold = 2.4
DR_layer = 4.0
layer_mapping = {
    1: {"n_layers": 1, "numWires": 47, "base_radius": 32.0},
    2: {"n_layers": 2, "numWires": 56, "base_radius": 38.0},
    3: {"n_layers": 2, "numWires": 72, "base_radius": 48.0},
    4: {"n_layers": 2, "numWires": 87, "base_radius": 58.0},
    5: {"n_layers": 1, "numWires": 99, "base_radius": 68.0},
}

# LUT ranges (should match your LUT file)
# Here we assume they were saved in the LUT file.
LUT, p_bins, theta_bins, phi_phase_bins = load_LUT()

# --- Simulation Functions (copied from your test/reco code) ---
def rotate_point(x, y, angle_deg):
    theta = math.radians(angle_deg)
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return xr, yr

def wire_position(superLayer, localLayer, wire):
    if superLayer not in layer_mapping:
        return (9999, 9999)
    mapping = layer_mapping[superLayer]
    numWires = mapping["numWires"]
    R_layer = mapping["base_radius"] + DR_layer * (localLayer - 1)
    alphaW = (2.0 * math.pi) / numWires
    angle = alphaW * (wire - 1) + 0.5 * math.radians(20.0)*(( -1)**(superLayer - 1))
    x = -R_layer * math.sin(angle)
    y = -R_layer * math.cos(angle)
    return (x, y)

def stereo_position(nominal_xy, stereo_offset_deg):
    theta = math.radians(stereo_offset_deg)
    x0, y0 = nominal_xy
    x = x0 * math.cos(theta) - y0 * math.sin(theta)
    y = x0 * math.sin(theta) + y0 * math.cos(theta)
    return (x, y)

def get_stereo_offset(superLayer, localLayer):
    stereo_sets = {1: [10], 2: [10, -10], 3: [10, -10], 4: [10, -10], 5: [10]}
    if superLayer in stereo_sets:
        offsets = stereo_sets[superLayer]
        if len(offsets)==1:
            return offsets[0]
        elif len(offsets)>=2:
            return offsets[0] if localLayer==1 else offsets[1]
    return 0.0

def point_to_segment_distance(P, A, B):
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
    theta_rad = theta  # theta is in radians
    pT = p * math.sin(theta_rad)
    R_val = (pT / (0.3 * B_field)) * 1000.0  # mm
    center_x = R_val * math.sin(0.0)
    center_y = -R_val * math.cos(0.0)
    x = center_x - R_val * math.sin(0.0 - t / R_val)
    y = center_y + R_val * math.cos(0.0 - t / R_val)
    z = z0 + t * math.cos(theta_rad)
    return np.array([x, y, z])

def compute_min_distance(P1, P2, p, theta, z0):
    theta_rad = theta  # theta already in radians
    t_max = (z_geom_far - z0) / math.cos(theta_rad)
    ts = np.linspace(0, t_max, 200)
    distances = [point_to_segment_distance(helix_point(t, theta_rad, z0, p, B_field, phi=0.0), P1, P2) for t in ts]
    return min(distances)

def simulate_hit_pattern(p, theta, phi_phase):
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
    return np.packbits(hit_pattern)

# --- Main: simulate events and save into ROOT TTree ---
N_events = 500

# Create a ROOT file and TTree
root_file = ROOT.TFile("AHDC_LUT_reco.root", "RECREATE")
tree = ROOT.TTree("tree", "LUT Reco Tree")

# Create branches (using array('f',[0]) to hold float values)
true_p  = array('f', [0])
true_theta  = array('f', [0])
true_phi  = array('f', [0])
reco_p  = array('f', [0])
reco_theta  = array('f', [0])
reco_phi  = array('f', [0])

tree.Branch("true_p", true_p, "true_p/F")
tree.Branch("true_theta", true_theta, "true_theta/F")
tree.Branch("true_phi", true_phi, "true_phi/F")
tree.Branch("reco_p", reco_p, "reco_p/F")
tree.Branch("reco_theta", reco_theta, "reco_theta/F")
tree.Branch("reco_phi", reco_phi, "reco_phi/F")

# Loop over events
for i in range(N_events):
    # Randomly generate true parameters within the LUT ranges.
    true_p[0] = random.uniform(p_bins[0], p_bins[-1])
    true_theta[0] = random.uniform(theta_bins[0], theta_bins[-1])
    # Use phi phase in [phi_phase_bins[0], phi_phase_bins[-1]) (e.g. 0 to 10 deg)
    true_phi[0] = random.uniform(phi_phase_bins[0], phi_phase_bins[-1])
    
    # Simulate hit pattern and pack it.
    hit_bool = simulate_hit_pattern(true_p[0], true_theta[0], true_phi[0])
    packed = pack_hit_pattern(hit_bool)
    
    # Run LUT reconstruction.
    (rp, rt, rphi), hamming = reconstruct_hit_pattern(packed, LUT, p_bins, theta_bins, phi_phase_bins)
    reco_p[0] = rp
    reco_theta[0] = rt
    reco_phi[0] = rphi

    tree.Fill()

# Write and close file.
tree.Write()
root_file.Close()
print("Saved", N_events, "events to AHDC_LUT_reco.root")
