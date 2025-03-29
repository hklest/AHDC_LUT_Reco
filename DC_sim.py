#!/usr/bin/env python3
"""
drift_chamber_sim.py

A standalone simulation of the drift chamber response to an incident track.

Left panel (2D endplate view):
  - Draws all AHDC wires (nominal positions at z = 0) as light gray circles.
  - Overlays hit wires (red) if the analytically computed minimum distance between 
    the wire cell (sampled along its finite segment) and the simulated track is 
    below threshold (here, 5 mm).

Right panel (3D view):
  - Draws the fixed chamber geometry: two endplates at z = 0 and z = 300 mm.
  - For each wire cell (each stereo‐set), draws a line connecting the near 
    (nominal) and far (stereo‐shifted) positions.
  - Colors a cell red if the analytic minimum distance between its segment and the 
    track is below threshold.
  - Draws the simulated track.
  
Usage:
   python3 drift_chamber_sim.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Global Parameters
# -------------------------
# Fixed chamber geometry (mm)
z_geom_near = 0.0         # near endplate (always at z = 0 for geometry)
z_geom_far = 300.0        # far endplate (300 mm)

# Track parameters
do_cosmics = False       # if True, track is drawn through the entire chamber; else, truncated.
track_z0 = 50.0          # track origin z (mm); for fixed target, this is > 0.
theta_deg = 45.0         # polar angle in degrees
phi_deg = 90.0           # azimuthal angle in degrees; for fixed target, phi=0 → track in +x.
p = 0.15                    # momentum in GeV/c (total momentum)
B_field = 5.0            # magnetic field in Tesla (solenoidal, +z direction)

# Stereo sets for AHDC wires (in degrees)
# For superlayers 1 and 5: only one set: +10.
# For superlayers 2-4: two sets: localLayer 1: +10, localLayer 2: -10.
stereo_sets = {1: [10], 2: [10, -10], 3: [10, -10], 4: [10, -10], 5: [10]}

# Hardcoded axis limits
xy_lim_2d = (-75, 75)
xy_lim_3d = (-100, 100)
z_lim_3d = (z_geom_near, z_geom_far+50)

hit_threshold = 2.4  # mm threshold for considering a wire "hit"

# Mapping for AHDC: superlayer -> (n_layers, numWires, base_radius)
layer_mapping = {
    1: {"n_layers": 1, "numWires": 47, "base_radius": 32.0},
    2: {"n_layers": 2, "numWires": 56, "base_radius": 38.0},
    3: {"n_layers": 2, "numWires": 72, "base_radius": 48.0},
    4: {"n_layers": 2, "numWires": 87, "base_radius": 58.0},
    5: {"n_layers": 1, "numWires": 99, "base_radius": 68.0},
}
DR_layer = 4.0  # mm increment per additional layer

# -------------------------
# Geometry Functions for AHDC
# -------------------------
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
    Rotate the nominal (x,y) position by the given stereo offset (degrees).
    """
    theta = math.radians(stereo_offset_deg)
    x0, y0 = nominal_xy
    x = x0 * math.cos(theta) - y0 * math.sin(theta)
    y = x0 * math.sin(theta) + y0 * math.cos(theta)
    return x, y

# -------------------------
# Track Analytical Distance Calculation
# -------------------------
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

def helix_point(t, theta, phi, z0, R, center_x, center_y):
    """
    Return the (x,y,z) point on the helix at path length t.
    The helix is parameterized as:
      x(t) = center_x - R * sin(phi - t/R)
      y(t) = center_y + R * cos(phi - t/R)
      z(t) = z0 + t * cos(theta)
    """
    x = center_x - R * math.sin(phi - t / R)
    y = center_y + R * math.cos(phi - t / R)
    z = z0 + t * math.cos(theta)
    return np.array([x, y, z])

def straight_point(t, theta, phi, z0):
    """
    Return the (x,y,z) point on a straight-line track at parameter t.
    """
    x = t * math.sin(theta) * math.cos(phi)
    y = t * math.sin(theta) * math.sin(phi)
    z = z0 + t * math.cos(theta)
    return np.array([x, y, z])

def compute_min_distance(P1, P2):
    """
    Compute the minimum distance between the finite wire segment [P1,P2]
    and the track.
    
    For B_field > 0, the track is a helix.
    For B_field == 0, the track is a straight line.
    
    This function uses a grid search over the parameter t (which is the path 
    length along the track) from t = 0 to t_max, where t_max is set by the 
    condition z = z_geom_far.
    """
    theta_val = math.radians(theta_deg)
    phi_val = math.radians(phi_deg)
    if B_field == 0.0:
        # Straight-line track.
        t_max = (z_geom_far - track_z0) / math.cos(theta_val)
        ts = np.linspace(0, t_max, 200)
        distances = []
        for t in ts:
            H = straight_point(t, theta_val, phi_val, track_z0)
            distances.append(point_to_segment_distance(H, P1, P2))
        return min(distances)
    else:
        # Helical track.
        # Calculate helix parameters.
        pT = p * math.sin(theta_val)  # transverse momentum
        R_val = (pT / (0.3 * B_field)) * 1000.0  # curvature radius in mm
        center_x = R_val * math.sin(phi_val)
        center_y = -R_val * math.cos(phi_val)
        # t_max is determined by z reaching z_geom_far:
        t_max = (z_geom_far - track_z0) / math.cos(theta_val)
        ts = np.linspace(0, t_max, 200)
        distances = []
        for t in ts:
            H = helix_point(t, theta_val, phi_val, track_z0, R_val, center_x, center_y)
            distances.append(point_to_segment_distance(H, P1, P2))
        return min(distances)

def compute_wire_distance(superLayer, localLayer, wire, stereo_offset_deg, z_near, z_far):
    """
    Compute the minimum distance between a wire cell and the track.
    The wire cell is defined as the segment from the nominal (near-endplate)
    position at z_near to the stereo-shifted (far-endplate) position at z_far.
    """
    pos_near = np.array(wire_position(superLayer, localLayer, wire))
    P1 = np.array([pos_near[0], pos_near[1], z_near])
    pos_far_nom = np.array(wire_position(superLayer, localLayer, wire))
    pos_far = np.array(stereo_position(pos_far_nom, stereo_offset_deg))
    P2 = np.array([pos_far[0], pos_far[1], z_far])
    return compute_min_distance(P1, P2)

# -------------------------
# 2D Endplate View (Left Panel)
# -------------------------
def plot_endplate(ax, stereo_sets, z_near=z_geom_near, z_far=z_geom_far, threshold=hit_threshold):
    """
    Draw all wires (nominal positions) on the near endplate (z = 0) in light gray.
    Overlay hit wires in red if the analytically computed minimum distance between 
    the wire cell and the track is below threshold.
    """
    # Draw all wires in light gray.
    for sl in range(1, 6):
        mapping = layer_mapping.get(sl)
        if mapping is None:
            continue
        n_layers = mapping["n_layers"]
        numWires = mapping["numWires"]
        for la in range(1, n_layers+1):
            for w in range(1, numWires+1):
                x, y = wire_position(sl, la, w)
                ax.plot(x, y, 'o', color='lightgray', markersize=4)
    hit_x, hit_y = [], []
    for sl in range(1, 6):
        mapping = layer_mapping.get(sl)
        if mapping is None:
            continue
        n_layers = mapping["n_layers"]
        numWires = mapping["numWires"]
        for la in range(1, n_layers+1):
            stereo_off = get_stereo_offset(sl, la)
            for w in range(1, numWires+1):
                d = compute_wire_distance(sl, la, w, stereo_off, z_near, z_far)
                if d < threshold:
                    x, y = wire_position(sl, la, w)
                    hit_x.append(x)
                    hit_y.append(y)
    ax.plot(hit_x, hit_y, 'ro', markersize=6)
    ax.set_xlim(-75,75)
    ax.set_ylim(-75,75)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("2D Endplate (z = 0) with Hit Wires")
    ax.set_aspect('equal')

# -------------------------
# 3D Geometry Visualization (Right Panel)
# -------------------------
def plot_3d_geometry(ax, stereo_offset_deg=10.0, z_near=z_geom_near, z_far=z_geom_far, threshold=hit_threshold):
    """
    Draw the 3D chamber geometry (wire cells) and the track.
    For each wire cell, draw the segment from the near (nominal) position to the far 
    (stereo-shifted) position, colored red if the minimum distance to the track is below threshold.
    Also, draw the track as a blue line.
    """
    # Draw fixed endplate outlines.
    theta_vals = np.linspace(0, 2*math.pi, 100)
    R_outline = 75
    x_circ = R_outline * np.cos(theta_vals)
    y_circ = R_outline * np.sin(theta_vals)
    ax.plot(x_circ, y_circ, zs=z_near, zdir='z', color='gray', alpha=0.5)
    ax.plot(x_circ, y_circ, zs=z_far, zdir='z', color='gray', alpha=0.5)
    
    # Draw the track.
    # (For display purposes, we use the already available compute_track_points function.)
    track_pts, _ = compute_track_points(num_points=1000)
    ax.plot(track_pts[:,0], track_pts[:,1], track_pts[:,2], color='blue', lw=2)
    
    # Draw each wire cell.
    for sl in range(1, 6):
        mapping = layer_mapping.get(sl)
        if mapping is None:
            continue
        n_layers = mapping["n_layers"]
        numWires = mapping["numWires"]
        for la in range(1, n_layers+1):
            stereo_off = get_stereo_offset(sl, la)
            for w in range(1, numWires+1):
                near_xy = np.array(wire_position(sl, la, w))
                far_xy = np.array(stereo_position(near_xy, stereo_off))
                P1 = np.array([near_xy[0], near_xy[1], z_near])
                P2 = np.array([far_xy[0], far_xy[1], z_far])
                d = compute_min_distance(P1, P2)
                if d < threshold:
                    col = 'red'
                    lw = 2.5
                else:
                    col = 'black'
                    lw = 0.5
                ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], color=col, lw=lw)
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(z_near, z_far+50)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D Chamber Geometry with Track")

# -------------------------
# Helper to get stereo offset from stereo_sets.
def get_stereo_offset(superLayer, localLayer):
    """
    Returns the stereo offset for a given superLayer and localLayer.
    For superlayers 1 and 5, returns +10.
    For superlayers 2-4, returns +10 for localLayer 1 and -10 for localLayer 2.
    """
    if superLayer in stereo_sets:
        offsets = stereo_sets[superLayer]
        if len(offsets) == 1:
            return offsets[0]
        elif len(offsets) >= 2:
            return offsets[0] if localLayer == 1 else offsets[1]
    return 0.0

# -------------------------
# Track Simulation (for display)
# -------------------------
def compute_track_points(num_points=100):
    """
    Compute a set of points along the track for display purposes.
    (This function is kept for visualization only.)
    """
    theta_val = math.radians(theta_deg)
    phi_val = math.radians(phi_deg)
    origin = np.array([0.0, 0.0, track_z0])
    if B_field == 0.0:
        d = np.array([math.sin(theta_val)*math.cos(phi_val),
                      math.sin(theta_val)*math.sin(phi_val),
                      math.cos(theta_val)])
        points = []
        t = 0
        t_max = (z_geom_far - track_z0) / d[2]
        dt = t_max/num_points
        while t < t_max:
            pt = origin + t * d
            points.append(pt)
            t += dt
        return np.array(points), d
    else:
        pT = p * math.sin(theta_val)
        R_val = (pT / (0.3 * B_field)) * 1000.0  # mm
        center_x = R_val * math.sin(phi_val)
        center_y = -R_val * math.cos(phi_val)
        points = []
        t_max = (z_geom_far - track_z0) / math.cos(theta_val)
        dt = t_max/num_points
        t = 0.0
        while t < t_max:
            x = center_x - R_val * math.sin(phi_val - t / R_val)
            y = center_y + R_val * math.cos(phi_val - t / R_val)
            z = track_z0 + t * math.cos(theta_val)
            points.append([x, y, z])
            t += dt
        return np.array(points), None

# -------------------------
# Main Visualization
# -------------------------
fig = plt.figure(figsize=(14,6))
ax_left = fig.add_subplot(1,2,1)
ax_right = fig.add_subplot(1,2,2, projection='3d')

# Plot 2D endplate view (Left Panel)
plot_endplate(ax_left, stereo_sets, z_near=z_geom_near, z_far=z_geom_far, threshold=hit_threshold)

# Plot 3D geometry view (Right Panel)
plot_3d_geometry(ax_right, stereo_offset_deg=10.0, z_near=z_geom_near, z_far=z_geom_far, threshold=hit_threshold)

plt.tight_layout()
plt.show()
