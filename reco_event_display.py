#!/usr/bin/env python3
"""
AHDC_LUT_event_display.py

Interactive event display: on pressing the "n" key, a new randomly generated track
is simulated, reconstructed via the LUT, and both the true and reconstructed tracks
are displayed (with the hit pattern overlay on a 2D endplate view and a 3D chamber view).

Usage:
  python3 AHDC_LUT_event_display.py
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LUT_reco import load_LUT, reconstruct_hit_pattern

# Global parameters (same as in the simulation)
z_geom_near = 0.0
z_geom_far = 300.0
track_z0   = 50.0
B_field    = 5.0
hit_threshold = 2.4
DR_layer   = 4.0
layer_mapping = {
    1: {"n_layers": 1, "numWires": 47, "base_radius": 32.0},
    2: {"n_layers": 2, "numWires": 56, "base_radius": 38.0},
    3: {"n_layers": 2, "numWires": 72, "base_radius": 48.0},
    4: {"n_layers": 2, "numWires": 87, "base_radius": 58.0},
    5: {"n_layers": 1, "numWires": 99, "base_radius": 68.0},
}

# --- Simulation functions (same as before) ---
def rotate_point(x, y, angle_deg):
    theta = math.radians(angle_deg)
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return xr, yr

def wire_position(superLayer, localLayer, wire):
    if superLayer not in layer_mapping:
        return 9999, 9999
    mapping = layer_mapping[superLayer]
    numWires = mapping["numWires"]
    R_layer = mapping["base_radius"] + DR_layer * (localLayer - 1)
    alphaW = (2.0 * math.pi) / numWires
    angle = alphaW * (wire - 1) + 0.5 * math.radians(20.0) * ((-1)**(superLayer-1))
    x = -R_layer * math.sin(angle)
    y = -R_layer * math.cos(angle)
    return x, y

def stereo_position(nominal_xy, stereo_offset_deg):
    theta = math.radians(stereo_offset_deg)
    x0, y0 = nominal_xy
    x = x0 * math.cos(theta) - y0 * math.sin(theta)
    y = x0 * math.sin(theta) + y0 * math.cos(theta)
    return x, y

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
    theta_rad = theta
    pT = p * math.sin(theta_rad)
    R_val = (pT / (0.3 * B_field)) * 1000.0
    center_x = R_val * math.sin(0.0)
    center_y = -R_val * math.cos(0.0)
    x = center_x - R_val * math.sin(0.0 - t / R_val)
    y = center_y + R_val * math.cos(0.0 - t / R_val)
    z = track_z0 + t * math.cos(theta_rad)
    return np.array([x, y, z])

def compute_min_distance(P1, P2, p, theta, z0):
    theta_rad = theta
    t_max = (z_geom_far - z0) / math.cos(theta_rad)
    ts = np.linspace(0, t_max, 200)
    distances = []
    for t in ts:
        H = helix_point(t, theta_rad, z0, p, B_field, phi=0.0)
        distances.append(point_to_segment_distance(H, P1, P2))
    return min(distances)

def simulate_hit_pattern(p, theta_deg, phi_phase):
    theta_rad = math.radians(theta_deg)
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
                d = compute_min_distance(P1, P2, p, math.radians(theta_deg), track_z0)
                hit_pattern.append(d < hit_threshold)
    return np.array(hit_pattern, dtype=bool)

def pack_hit_pattern(hit_pattern):
    return np.packbits(hit_pattern)

# --- Load LUT ---
LUT, p_bins_LUT, theta_bins_LUT, phi_phase_bins = load_LUT()

def generate_event():
    # Generate a random true event within LUT ranges
    true_p = random.uniform(p_bins_LUT[0], p_bins_LUT[-1])
    true_theta = random.uniform(theta_bins_LUT[0], theta_bins_LUT[-1])
    true_phi = random.uniform(phi_phase_bins[0], phi_phase_bins[-1])
    hit_bool = simulate_hit_pattern(true_p, true_theta, true_phi)
    packed = pack_hit_pattern(hit_bool)
    (reco_p, reco_theta, reco_phi), dist = reconstruct_hit_pattern(packed, LUT, p_bins_LUT, theta_bins_LUT, phi_phase_bins)
    return true_p, true_theta, true_phi, reco_p, reco_theta, reco_phi

# --- Plotting the chamber and tracks (2D and 3D) ---
def plot_event(true_params, reco_params):
    true_p, true_theta, true_phi = true_params
    reco_p, reco_theta, reco_phi = reco_params
    
    # For visualization, compute track points (we use helix_point)
    theta_rad_true = math.radians(true_theta)
    theta_rad_reco = math.radians(reco_theta)
    t_max = (z_geom_far - track_z0) / math.cos(theta_rad_true)
    ts = np.linspace(0, t_max, 300)
    true_track = np.array([helix_point(t, theta_rad_true, track_z0, true_p, B_field) for t in ts])
    reco_track = np.array([helix_point(t, theta_rad_reco, track_z0, reco_p, B_field) for t in ts])
    
    # Create figure with 2 subplots: 2D endplate view and 3D view.
    fig = plt.figure(figsize=(14,6))
    
    # 2D Endplate view: plot near-endplate positions of wires and overlay hit wires
    ax2d = fig.add_subplot(1,2,1)
    # Plot all wires (light gray)
    for sl in sorted(layer_mapping.keys()):
        mapping = layer_mapping[sl]
        n_layers = mapping["n_layers"]
        numWires = mapping["numWires"]
        for la in range(1, n_layers+1):
            for w in range(1, numWires+1):
                x, y = wire_position(sl, la, w)
                # Apply chamber rotation using true_phi (for visualization)
                x, y = rotate_point(x, y, true_phi)
                ax2d.plot(x, y, 'o', color='lightgray', markersize=2)
    # Overlay true hit wires in blue and reco hit wires in red (for simplicity we reuse true_phi for both)
    # In a more complete version you might recompute the hit pattern for each.
    ax2d.plot(true_track[:,0], true_track[:,1], '-', color='blue', label="True Track")
    ax2d.plot(reco_track[:,0], reco_track[:,1], '--', color='red', label="Reco Track")
    ax2d.set_title("2D Endplate View (z=0)")
    ax2d.set_xlabel("x (mm)")
    ax2d.set_ylabel("y (mm)")
    ax2d.legend()
    ax2d.set_aspect('equal')
    
    # 3D view: plot chamber outline and tracks
    ax3d = fig.add_subplot(1,2,2, projection='3d')
    # Draw chamber endplates as circles
    theta_vals = np.linspace(0, 2*math.pi, 100)
    R_outline = 75
    x_circ = R_outline * np.cos(theta_vals)
    y_circ = R_outline * np.sin(theta_vals)
    ax3d.plot(x_circ, y_circ, zs=z_geom_near, zdir='z', color='gray', alpha=0.5)
    ax3d.plot(x_circ, y_circ, zs=z_geom_far, zdir='z', color='gray', alpha=0.5)
    ax3d.plot(true_track[:,0], true_track[:,1], true_track[:,2], color='blue', lw=2, label="True")
    ax3d.plot(reco_track[:,0], reco_track[:,1], reco_track[:,2], color='red', lw=2, label="Reco")
    ax3d.set_title("3D Chamber View")
    ax3d.set_xlabel("x (mm)")
    ax3d.set_ylabel("y (mm)")
    ax3d.set_zlabel("z (mm)")
    ax3d.legend()
    
    # Display true and reco parameter values as text
    fig.suptitle(f"True: p={true_p:.3f} GeV/c, θ={true_theta:.2f}°, φ={true_phi:.2f}°   |   Reco: p={reco_p:.3f} GeV/c, θ={reco_theta:.2f}°, φ={reco_phi:.2f}°")
    
    plt.tight_layout()
    plt.show()

# --- Interactive event display ---
def on_key(event):
    if event.key == "n":
        true_p_val, true_theta_val, true_phi_val, reco_p_val, reco_theta_val, reco_phi_val = generate_event()
        print(f"True: p={true_p_val:.3f}, theta={true_theta_val:.2f}, phi={true_phi_val:.2f}  |  Reco: p={reco_p_val:.3f}, theta={reco_theta_val:.2f}, phi={reco_phi_val:.2f}")
        plot_event((true_p_val, true_theta_val, true_phi_val),
                   (reco_p_val, reco_theta_val, reco_phi_val))

# Create an initial event display and connect key press.
fig = plt.figure()
fig.canvas.mpl_connect("key_press_event", on_key)
print("Press 'n' in the plot window to display a new event.")
plt.show()
