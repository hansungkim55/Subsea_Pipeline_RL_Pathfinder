# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:13:53 2025

Author: HanSung Kim

Subsea Pipeline Route Planning via Double Dueling DQN (TensorFlow 2.x)

Assumptions:
- The pipe follows the seabed surface (SEABED_MODE=True).
- User input: Start/Goal are (x, y) in meters; risk zones are list of ((cx, cy), radius[m]).
- Risk zones are vertical cylinders from seabed (k=seabed_k) to sea surface (k=0).
- OBSU in [0,2]; OBSU >= 1.0 is forbidden. OBSU == 0 handled by soft penalty (not hard-blocked).
- Reward terms are meter/degree-scaled; final Top-3 routes are ranked by engineering KPIs.

Visual outputs:
- Episode XY plots, optional 3D previews, final 3D route.
"""

#%% Headers, performance settings, hyperparameters

# CPU parallel hints (set before TF import)
import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')   # no pop-up during training
import matplotlib.pyplot as plt
from collections import deque
import math
import random
from copy import deepcopy

# Output directory cleanup (optional)
target = r"C:\\Users\\goldw\\Downloads\\ML_Engineer\\episode_frames"
try:
    if os.path.isdir(target):
        for name in os.listdir(target):
            p = os.path.join(target, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                print(f"Delete failed: {p} -> {e}")
except Exception:
    pass

# Threading knobs for BLAS/TF
_num_threads = str(os.cpu_count() or 12)
os.environ["OMP_NUM_THREADS"] = _num_threads
os.environ["MKL_NUM_THREADS"] = _num_threads
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

# TF threading / XLA
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(int(_num_threads))
tf.config.optimizer.set_jit(True)

# ----------------------------
# User settings
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# Grid and export
DX, DY, DZ = 5, 5, 5                 # [m]
NX, NY, NZ = 400, 400, 300
EXPORT_DS_M = 12.3                   # export spacing [m]

USER_START_M = (250.0, 300.0)
USER_GOAL_M  = (1800.0, 1750.0)

RISK_ZONES = [((700.0, 700.0), 40.0),
              ((1250.0, 1250.0), 100.0)]

OBSU_FORBID_EQUAL_ZERO = False
OBSU_FORBID_GE = 1.0

# RL
EPISODES = 2000
BATCH_SIZE = 512
TARGET_UPDATE_STEPS = 5000
GAMMA = 0.99
REPLAY_CAPACITY = 1_000_000
LEARNING_RATE = 3e-4

# Reward weights (meter/degree basis)
REWARD_GOAL = 1000.0
REWARD_WAYPOINT = 120.0

W_LENGTH_PER_M   = -0.04
C_PROGRESS_PER_M = +0.05
K_BEND   = 1.5
K_CLEAR  = 4.0
K_OBSU   = 4.0
K_SPAN   = 6.0
K_SLOPE  = 3.0

TURN_RADIUS_MIN_M      = 650.0
TURN_RADIUS_PENALTY_W  = -25.0
USE_TURN_HARD_BLOCK    = True

USE_FREESPAN_HARD_BLOCK = False
FREESPAN_HARD_LIMIT_M   = 60.0

RISK_CLEARANCE_M  = 120.0

LINE_CORRIDOR_M   = 260.0
LINE_BONUS_WEIGHT = +0.9
USE_LINE_SHAPING  = False

GOAL_TOL_WIDE_INIT  = 120.0
GOAL_TOL_BASE_INIT  = 60.0
GOAL_TOL_WIDE_FINAL = 50.0
GOAL_TOL_BASE_FINAL = 30.0
CURRICULUM_FRAC = 0.8

WP_SPACING_M = 90.0
WP_TOL_M     = 40.0

N_ENVS = max(8, os.cpu_count() or 8)

# Free-span threshold [m]
gap_thresh = 0.15


#%% Coordinates, terrain, masks

def meters_to_ij(x_m, y_m):
    i = int(round(x_m / DX))
    j = int(round(y_m / DY))
    return np.clip(i, 0, NX-1), np.clip(j, 0, NY-1)

def idx_to_meters(i, j, k):
    return i * DX, j * DY, k * DZ

def make_seabed(nx=NX, ny=NY, dx=DX, dy=DY, base_depth=800.0):
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X2D, Y2D = np.meshgrid(x, y, indexing='ij')
    z = (base_depth
         + 80.0*np.sin(2*np.pi*X2D/3000.0)
         + 60.0*np.sin(2*np.pi*Y2D/2200.0)
         + 30.0*np.sin(2*np.pi*(X2D+Y2D)/1800.0))
    rng = np.random.default_rng(SEED)
    for _ in range(6):
        cx = rng.uniform(0, x[-1]); cy = rng.uniform(0, y[-1])
        r  = rng.uniform(80, 160)
        h  = rng.uniform(-6.0, 8.0)
        d2 = (X2D-cx)**2 + (Y2D-cy)**2
        z += h*np.exp(-d2/(2*r*r))
    ZMAX = (NZ-1) * DZ
    z = np.clip(z, 200.0, ZMAX - DZ)
    return z.astype(np.float32)

SEABED_Z = make_seabed()
seabed_k = np.clip(np.round(SEABED_Z / DZ).astype(int), 0, NZ-1)

# OBSU field (~0.1%–1% high risk >=1.0 for demo)
p_high = 0.001
mask_high = (np.random.rand(NX, NY, NZ) < p_high)
obsu_low  = np.random.uniform(0.0, 1.0, size=(NX, NY, NZ)).astype(np.float32)
obsu_high = np.random.uniform(1.0, 2.0, size=(NX, NY, NZ)).astype(np.float32)
OBSU = np.where(mask_high, obsu_high, obsu_low).astype(np.float32)
print(f"OBSU >= 1 ratio ≈ {100*np.mean(OBSU >= 1.0):.2f}%")

# Forbidden by OBSU
obsu_forbid = np.zeros((NX,NY,NZ), dtype=bool)
if OBSU_FORBID_EQUAL_ZERO:
    obsu_forbid |= (OBSU == 0.0)
obsu_forbid |= (OBSU >= OBSU_FORBID_GE)

XX, YY = np.meshgrid(np.arange(NX)*DX, np.arange(NY)*DY, indexing='ij')

def build_risk_mask_cylinders():
    risk = np.zeros((NX,NY,NZ), dtype=bool)
    K  = np.arange(NZ)[None, None, :]
    SK = seabed_k[:, :, None]
    for (cx, cy), radius in RISK_ZONES:
        d2 = (XX - cx)**2 + (YY - cy)**2
        within_xy = (d2 <= radius*radius)
        cyl = within_xy[:, :, None] & (K <= SK)
        risk |= cyl
    return risk

RISK_MASK = build_risk_mask_cylinders()
INVALID_MASK = RISK_MASK | obsu_forbid

# 2D projections for plotting
risk_xy = np.zeros((NX, NY), dtype=bool)
for (cx, cy), radius in RISK_ZONES:
    d2 = (XX - cx)**2 + (YY - cy)**2
    risk_xy |= (d2 <= radius*radius)
obsu_only_xy = obsu_forbid[np.arange(NX)[:, None], np.arange(NY)[None, :], seabed_k] & (~risk_xy)
invalid_xy = INVALID_MASK[np.arange(NX)[:, None], np.arange(NY)[None, :], seabed_k]

def nearest_valid_xy_on_seabed(i, j, invalid_mask, seabed_k_map):
    kk = seabed_k_map[i, j]
    if not invalid_mask[i, j, kk]:
        return (i, j, kk)
    for r in range(1, max(NX, NY)):
        imin, imax = max(0, i-r), min(NX-1, i+r)
        jmin, jmax = max(0, j-r), min(NY-1, j+r)
        for ii in (imin, imax):
            for jj in range(jmin, jmax+1):
                kk = seabed_k_map[ii, jj]
                if not invalid_mask[ii, jj, kk]:
                    return (ii, jj, kk)
        for jj in (jmin, jmax):
            for ii in range(imin, imax+1):
                kk = seabed_k_map[ii, jj]
                if not invalid_mask[ii, jj, kk]:
                    return (ii, jj, kk)
    raise RuntimeError("No valid seabed cell found nearby.")

si, sj = meters_to_ij(*USER_START_M)
gi, gj = meters_to_ij(*USER_GOAL_M)
START_IDX = nearest_valid_xy_on_seabed(si, sj, INVALID_MASK, seabed_k)
GOAL_IDX  = nearest_valid_xy_on_seabed(gi, gj, INVALID_MASK, seabed_k)

def _point_to_segment_dist(px, py, ax, ay, bx, by):
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx*abx + aby*aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
    cx, cy = ax + t*abx, ay + t*aby
    return math.hypot(px - cx, py - cy)

si, sj, _ = START_IDX
gi, gj, _ = GOAL_IDX
sx_m, sy_m = si*DX, sj*DY
gx_m, gy_m = gi*DX, gj*DY

LINE_DIST = np.zeros((NX, NY), dtype=np.float32)
for i in range(NX):
    xi = i*DX
    for j in range(NY):
        yj = j*DY
        LINE_DIST[i, j] = _point_to_segment_dist(xi, yj, sx_m, sy_m, gx_m, gy_m)

def build_risk_edge_distance():
    D = np.full((NX, NY), np.inf, dtype=np.float32)
    for (cx, cy), radius in RISK_ZONES:
        d = np.sqrt((XX - cx)**2 + (YY - cy)**2) - radius
        D = np.minimum(D, d)
    return D

RISK_EDGE_DIST = build_risk_edge_distance()

# Slope (deg)
dZdx, dZdy = np.gradient(SEABED_Z, DX, DY)
SLOPE_DEG = np.degrees(np.arctan(np.sqrt(dZdx**2 + dZdy**2)))
SLOPE_MAX_DEG = 10.0


#%% Waypoints, preview, plotting, export

def build_waypoints(sx, sy, gx, gy, spacing_m=WP_SPACING_M):
    seg_len = math.hypot(gx - sx, gy - sy)
    n = max(1, int(seg_len // spacing_m))
    xs = np.linspace(sx, gx, n+1)
    ys = np.linspace(sy, gy, n+1)
    pts = [(int(round(x/DX)), int(round(y/DY))) for x, y in zip(xs[1:-1], ys[1:-1])]
    pts = [(min(max(i,0),NX-1), min(max(j,0),NY-1)) for (i,j) in pts]
    return pts

WAYPOINTS = build_waypoints(sx_m, sy_m, gx_m, gy_m)

def visualize_space_preview_3d(max_points_per_cloud=2000, show_invalid=True,
                               draw_seabed=True, stride=4, surface=True):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = np.arange(0, NX, 4) * DX
    ys = np.arange(0, NY, 4) * DY
    Xs, Ys = np.meshgrid(xs, ys, indexing='ij')
    Zs = SEABED_Z[::4, ::4]
    ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, linewidth=0.4, alpha=0.7)

    def _sample(mask, max_pts):
        idx = np.where(mask); n = idx[0].size
        if n == 0: return None
        if n > max_pts:
            sel = np.random.choice(n, size=max_pts, replace=False)
            ix, iy, iz = idx[0][sel], idx[1][sel], idx[2][sel]
        else:
            ix, iy, iz = idx
        return ix*DX, iy*DY, iz*DZ

    pts_risk = _sample(RISK_MASK, max_points_per_cloud)
    if pts_risk is not None:
        ax.scatter(*pts_risk, s=2, alpha=0.35, color='black', label='Risk Zones')

    if show_invalid:
        mask_surface_invalid = np.zeros((NX, NY), dtype=bool)
        for i in range(NX):
            for j in range(NY):
                k = seabed_k[i, j]
                if INVALID_MASK[i, j, k] and not RISK_MASK[i, j, k]:
                    mask_surface_invalid[i, j] = True
        idx = np.where(mask_surface_invalid)
        if idx[0].size > 0:
            xs, ys = idx[0]*DX, idx[1]*DY
            zs = SEABED_Z[idx]
            ax.scatter(xs, ys, zs, s=4, alpha=0.6, color='orange', label='OBSU invalid')

    si, sj, sk = START_IDX
    gi, gj, gk = GOAL_IDX
    sx, sy, sz = si*DX, sj*DY, SEABED_Z[si, sj]
    gx, gy, gz = gi*DX, gj*DY, SEABED_Z[gi, gj]
    ax.scatter([sx],[sy],[sz], s=80, marker='o', c='green', label='Start')
    ax.scatter([gx],[gy],[gz], s=80, marker='^', c='red', label='Goal')

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Depth [m]')
    ax.set_title('3D Space Preview')
    ax.invert_zaxis()
    try:
        ax.set_box_aspect((NX*DX, NY*DY, 500))
    except Exception:
        pass
    ax.legend(loc='upper left')
    plt.show()

SHOW_SPACE_PREVIEW = False
if SHOW_SPACE_PREVIEW:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    visualize_space_preview_3d()
    print("Preview shown. Set SHOW_SPACE_PREVIEW=False to continue.")
    sys.exit()

def _path_length_m_from_idx(path_idx_seq):
    if len(path_idx_seq) < 2: return 0.0
    total = 0.0
    for (i1,j1,k1),(i2,j2,k2) in zip(path_idx_seq[:-1], path_idx_seq[1:]):
        x1,y1,z1 = i1*DX, j1*DY, SEABED_Z[i1,j1]
        x2,y2,z2 = i2*DX, j2*DY, SEABED_Z[i2,j2]
        total += math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return total

def plot_episode_xy(path_idx_seq, episode, save_dir="episode_frames", show=False, with_contour=False,
                    ep_return=None, ep_eps=None):
    os.makedirs(save_dir, exist_ok=True)
    xs = [i*DX for (i,j,k) in path_idx_seq]
    ys = [j*DY for (i,j,k) in path_idx_seq]

    si, sj, _ = START_IDX
    gi, gj, _ = GOAL_IDX
    sx, sy = si*DX, sj*DY
    gx, gy = gi*DX, gj*DY

    plt.figure(figsize=(6,6)); ax = plt.gca()
    ax.imshow(risk_xy.T, origin='lower',
              extent=[0, NX*DX, 0, NY*DY], alpha=0.25, cmap='gray_r', interpolation='nearest')

    obsi = np.where(obsu_only_xy.T)
    if obsi[0].size > 0:
        yy = obsi[0] * DY
        xx = obsi[1] * DX
        ax.scatter(xx, yy, s=4, c='orange', alpha=0.7, label='OBSU invalid')

    if USE_LINE_SHAPING:
        tube_mask = (LINE_DIST < LINE_CORRIDOR_M)
        ax.imshow(tube_mask.T, origin='lower',
                  extent=[0, NX*DX, 0, NY*DY], alpha=0.10, cmap='Blues', interpolation='nearest')

    if len(WAYPOINTS) > 0:
        wx = [i*DX for (i,j) in WAYPOINTS]
        wy = [j*DY for (i,j) in WAYPOINTS]
        ax.scatter(wx, wy, s=50, edgecolors='blue', facecolors='none',
                   marker='o', linewidths=1.5, alpha=0.9, label='Waypoints')

    if with_contour:
        ax.contour(XX, YY, SEABED_Z, levels=20, linewidths=0.5, alpha=0.5)

    ax.plot(xs, ys, linewidth=2.0, label=f'Episode {episode} Path')
    ax.scatter([sx], [sy], s=60, c='green', marker='o', label='Start')
    ax.scatter([gx], [gy], s=60, c='red', marker='^', label='Goal')

    ax.set_xlim(0, NX*DX); ax.set_ylim(0, NY*DY)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Episode {episode} (XY)')

    path_len_m = _path_length_m_from_idx(path_idx_seq)
    annot = f"Path length: {path_len_m:.1f} m"
    if ep_return is not None:
        annot += f"\nReturn: {ep_return:.1f}"
    if ep_eps is not None:
        annot += f"\neps: {ep_eps:.3f}"
    ax.text(0.02, 0.98, annot, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=9)

    ax.legend(loc='lower right')
    out_path = os.path.join(save_dir, f'ep_{episode:04d}.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print(f"[viz] saved {out_path}")

def chaikin_smooth_xy(points_xy, n_iter=2):
    pts = np.array(points_xy, dtype=np.float32)
    for _ in range(n_iter):
        new_pts = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            Q = 0.75 * a + 0.25 * b
            R = 0.25 * a + 0.75 * b
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])
        pts = np.array(new_pts)
    return pts

def save_best_path_xyz_dense(path_idx_seq, filename, ds=EXPORT_DS_M):
    if not path_idx_seq or len(path_idx_seq) < 2:
        print(f"[export] path too short: {len(path_idx_seq)}"); return
    out = []
    for (i0, j0, k0), (i1, j1, k1) in zip(path_idx_seq[:-1], path_idx_seq[1:]):
        x0, y0 = i0 * DX, j0 * DY
        x1, y1 = i1 * DX, j1 * DY
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(L / max(ds, 1e-6))))
        for t in range(n):
            a = t / float(n)
            xt = x0 + a * (x1 - x0)
            yt = y0 + a * (y1 - y0)
            ii, jj = meters_to_ij(xt, yt)
            zt = float(SEABED_Z[ii, jj])
            out.append((xt, yt, zt))
    il, jl, kl = path_idx_seq[-1]
    out.append((il * DX, jl * DY, float(SEABED_Z[il, jl])))
    with open(filename, "w", encoding="utf-8") as f:
        for x, y, z in out:
            f.write(f"{x:.6f},{y:.6f},{z:.6f}\n")
    print(f"[export] saved dense path: {filename} | points={len(out)} | ds={ds} m")


#%% Action model, environment

DELTA_PSI_DEG = np.array([-5.0, 0.0, +5.0], dtype=np.float32)
STEP_M = 150

STEP_M_BIG   = 150.0
STEP_M_SMALL = 25.0

SAMPLE_DS = min(DX, DY)
NUM_ACTIONS = 3

def _wrap_angle(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def segment_intersects_circle(x0, y0, x1, y1, cx, cy, r):
    vx, vy = x1 - x0, y1 - y0
    wx, wy = x0 - cx, y0 - cy
    a = vx * vx + vy * vy
    b = 2.0 * (vx * wx + vy * wy)
    c = wx * wx + wy * wy - r * r
    t = 0.0
    if a > 1e-12:
        t = max(0.0, min(1.0, -b / (2.0 * a)))
    dx = (x0 + t * vx) - cx
    dy = (y0 + t * vy) - cy
    return (dx * dx + dy * dy) <= r * r

def goal_tolerance_by_ep(ep):
    ramp_start = int(CURRICULUM_FRAC * EPISODES)
    ramp_end   = EPISODES
    if ep <= ramp_start:
        return GOAL_TOL_WIDE_INIT, GOAL_TOL_BASE_INIT
    elif ep >= ramp_end:
        return GOAL_TOL_WIDE_FINAL, GOAL_TOL_BASE_FINAL
    else:
        alpha = (ep - ramp_start) / max(1, ramp_end - ramp_start)
        wide = (1 - alpha) * GOAL_TOL_WIDE_INIT + alpha * GOAL_TOL_WIDE_FINAL
        base = (1 - alpha) * GOAL_TOL_BASE_INIT + alpha * GOAL_TOL_BASE_FINAL
        return wide, base

class SubseaRouteEnv:
    def __init__(self, invalid_mask, obsu, start_idx, goal_idx, max_steps=600):
        self.invalid = invalid_mask
        self.obsu = obsu
        self.start = start_idx
        self.goal = goal_idx
        self.max_steps = max_steps
        self.current_episode = 0
        self.steps = 0
        self.path = []
        self.curr_span_len = 0.0
        self.reset()

    def reset(self):
        si, sj, sk = self.start
        self.x_m = si * DX; self.y_m = sj * DY
        gi, gj, gk = self.goal
        gx, gy = gi * DX, gj * DY
        self.heading = math.atan2(gy - self.y_m, gx - self.x_m)
        self.steps = 0
        self.path = [(si, sj, seabed_k[si, sj])]
        self.curr_span_len = 0.0
        return self._state()

    def _state(self):
        i, j = meters_to_ij(self.x_m, self.y_m)
        k = seabed_k[i, j]
        gi, gj, gk = self.goal
        spsi, cpsi = math.sin(self.heading), math.cos(self.heading)
        return np.array([i/(NX-1), j/(NY-1), k/(NZ-1),
                         gi/(NX-1), gj/(NY-1), gk/(NZ-1),
                         spsi, cpsi], dtype=np.float32)

    def _sample_cells_along(self, x0, y0, x1, y1):
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist / SAMPLE_DS)))
        cells = []
        for t in range(1, n + 1):
            alpha = t / n
            xt = x0 + alpha * (x1 - x0)
            yt = y0 + alpha * (y1 - y0)
            i, j = meters_to_ij(xt, yt)
            if i < 0 or i >= NX or j < 0 or j >= NY:
                return cells, True
            k = seabed_k[i, j]
            cells.append((i, j, k))
        return cells, False

    def _circumradius_m_xy(self, p0, p1, p2):
        (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
        a = math.hypot(x1 - x0, y1 - y0)
        b = math.hypot(x2 - x1, y2 - y1)
        c = math.hypot(x2 - x0, y2 - y0)
        area2 = abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        if area2 < 1e-9:
            return float('inf')
        return (a * b * c) / (2.0 * area2)

    def valid_action_mask(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for a_idx, ddeg in enumerate(DELTA_PSI_DEG):
            new_heading = _wrap_angle(self.heading + math.radians(ddeg))
            x1 = self.x_m + STEP_M * math.cos(new_heading)
            y1 = self.y_m + STEP_M * math.sin(new_heading)
            cells, oob = self._sample_cells_along(self.x_m, self.y_m, x1, y1)
            if oob or len(cells) == 0:
                mask[a_idx] = False
                continue
            ok = True
            for (ii, jj, kk) in cells:
                if kk < 0 or kk >= NZ or self.invalid[ii, jj, kk]:
                    ok = False; break
            mask[a_idx] = ok
        return mask

    def step(self, action_idx):
        self.steps += 1

        # candidate move
        dpsi = math.radians(float(DELTA_PSI_DEG[action_idx]))
        new_heading = _wrap_angle(self.heading + dpsi)
        x0, y0 = self.x_m, self.y_m
        x1 = x0 + STEP_M * math.cos(new_heading)
        y1 = y0 + STEP_M * math.sin(new_heading)

        # validity check
        cells, oob = self._sample_cells_along(x0, y0, x1, y1)
        if oob or len(cells) == 0:
            return self._state(), -600.0, True, {"violation": True}
        for (ii, jj, kk) in cells:
            if kk < 0 or kk >= NZ or self.invalid[ii, jj, kk]:
                return self._state(), -600.0, True, {"violation": True}

        step_len = STEP_M

        # ----- (1) length / progress (meters) -----
        gx_m, gy_m = self.goal[0] * DX, self.goal[1] * DY
        prev_goal_dist_m = math.hypot(gx_m - x0, gy_m - y0)
        new_goal_dist_m  = math.hypot(gx_m - x1, gy_m - y1)
        progress_m = max(0.0, prev_goal_dist_m - new_goal_dist_m)
        r_length   = W_LENGTH_PER_M * step_len
        r_progress = C_PROGRESS_PER_M * progress_m

        # ----- (2) curvature / MBR -----
        r_bend = -K_BEND * (abs(math.degrees(dpsi)) / 5.0) ** 2
        if len(self.path) >= 2:
            p0_i, p0_j, _ = self.path[-2]
            p1_i, p1_j, _ = self.path[-1]
            p0 = (p0_i * DX, p0_j * DY)
            p1 = (p1_i * DX, p1_j * DY)
            p2 = (x1, y1)
            R = self._circumradius_m_xy(p0, p1, p2)
            if R < TURN_RADIUS_MIN_M:
                if USE_TURN_HARD_BLOCK:
                    return self._state(), -600.0, True, {"violation": True, "reason": "min_turn_radius"}
                r_bend += TURN_RADIUS_PENALTY_W * (1.0 - R / max(TURN_RADIUS_MIN_M, 1e-6))

        # ----- (3) OBSU (soft) -----
        obsu_vals = [float(self.obsu[ii, jj, kk]) for (ii, jj, kk) in cells]
        max_obsu  = max(obsu_vals) if obsu_vals else 0.0
        r_obsu = -K_OBSU * max(0.0, (max_obsu - 0.6))

        # ----- (4) free span -----
        new_span_len = self.curr_span_len
        for c0, c1 in zip(cells[:-1], cells[1:]):
            (i, j, _), (ii, jj, _) = c0, c1
            if abs(ii - i) <= 1 and abs(jj - j) <= 1:
                mid_z = SEABED_Z[(i + ii) // 2, (j + jj) // 2]
                avg_z = 0.5 * (SEABED_Z[i, j] + SEABED_Z[ii, jj])
                gap = max(0.0, avg_z - mid_z)
            else:
                gap = 0.0
            segL = math.hypot((ii - i) * DX, (jj - j) * DY)
            if gap > gap_thresh:
                new_span_len += segL
            else:
                new_span_len = 0.0
        self.curr_span_len = new_span_len
        L_ALLOW = 20.0
        r_span = -K_SPAN * max(0.0, (new_span_len - L_ALLOW) / L_ALLOW)
        if USE_FREESPAN_HARD_BLOCK and new_span_len > FREESPAN_HARD_LIMIT_M:
            return self._state(), -600.0, True, {"violation": True, "reason": "free_span_limit"}

        # ----- (5) risk boundary clearance -----
        cur_i, cur_j = meters_to_ij(x0, y0)
        new_i, new_j = meters_to_ij(x1, y1)
        d_prev = float(RISK_EDGE_DIST[cur_i, cur_j])
        d_new  = float(RISK_EDGE_DIST[new_i, new_j])
        r_clear = 0.0
        if d_new >= 0.0:
            if d_new < RISK_CLEARANCE_M:
                r_clear += -K_CLEAR * (1.0 - d_new / max(RISK_CLEARANCE_M, 1e-6)) ** 2
            delta = d_new - max(0.0, d_prev)
            r_clear += min((delta / max(RISK_CLEARANCE_M, 1e-6)) * 0.5, 1.0)

        # ----- (6) slope -----
        slope_deg = float(SLOPE_DEG[new_i, new_j])
        r_slope = 0.0
        if slope_deg > SLOPE_MAX_DEG:
            r_slope = -K_SLOPE * ((slope_deg - SLOPE_MAX_DEG) / SLOPE_MAX_DEG) ** 2

        # ----- (7) goal -----
        wide, base = goal_tolerance_by_ep(self.current_episode)
        tol = wide if self.current_episode < int(CURRICULUM_FRAC * EPISODES) else base
        hit = segment_intersects_circle(x0, y0, x1, y1, gx_m, gy_m, tol)
        goal_xy_dist_m = math.hypot((new_i - self.goal[0]) * DX, (new_j - self.goal[1]) * DY)
        r_goal = 0.0
        done = False
        if hit or (goal_xy_dist_m <= tol):
            done = True
            r_goal = REWARD_GOAL

        # state update
        self.x_m, self.y_m = x1, y1
        self.heading = new_heading
        self.path.append((new_i, new_j, seabed_k[new_i, new_j]))
        if self.steps >= self.max_steps:
            done = True

        # total reward
        reward = r_length + r_progress + r_bend + r_obsu + r_span + r_clear + r_slope + r_goal
        info = {
            "violation": False,
            "components": {
                "r_length": float(r_length),
                "r_progress": float(r_progress),
                "r_bend": float(r_bend),
                "r_obsu": float(r_obsu),
                "r_span": float(r_span),
                "r_clear": float(r_clear),
                "r_slope": float(r_slope),
                "r_goal": float(r_goal),
                "step_len_m": float(step_len),
            },
        }
        return self._state(), reward, done, info


#%% Dueling DQN, replay, training

def build_dueling_dqn(input_dim, num_actions):
    inputs = keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(384, activation='relu')(inputs)
    x = layers.Dense(384, activation='relu')(x)
    v = layers.Dense(1, activation=None)(x)
    a = layers.Dense(num_actions, activation=None)(x)
    a_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
    a_centered = layers.Subtract()([a, a_mean])
    q = layers.Add()([v, a_centered])
    model = keras.Model(inputs=inputs, outputs=q)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.Huber()
    )
    return model

q_net = build_dueling_dqn(8, NUM_ACTIONS)
target_net = build_dueling_dqn(8, NUM_ACTIONS)
target_net.set_weights(q_net.get_weights())

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d, mask_next):
        self.buf.append((s, a, r, s2, d, mask_next))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d, m = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int32),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32),
                np.array(m, dtype=np.bool_))
    def __len__(self):
        return len(self.buf)

memory = ReplayBuffer(REPLAY_CAPACITY)
print('Training start!\n')

def current_step_m(ep):
    return STEP_M_BIG if ep < int(CURRICULUM_FRAC * EPISODES) else STEP_M_SMALL

L_xy = math.hypot((GOAL_IDX[0]-START_IDX[0])*DX, (GOAL_IDX[1]-START_IDX[1])*DY)

def set_episode_dynamics(ep, envs):
    global STEP_M, DELTA_PSI_DEG
    STEP_M = float(current_step_m(ep))

    SAFETY = 1.0
    max_delta_deg = np.degrees(SAFETY * STEP_M / max(TURN_RADIUS_MIN_M, 1e-9))
    max_delta_deg = float(np.clip(max_delta_deg, 0.5, 7.5))
    DELTA_PSI_DEG = np.array([-max_delta_deg, 0.0, +max_delta_deg], dtype=np.float32)

    H_nom = max(1, int(np.ceil(L_xy / max(STEP_M, 1e-9))))
    MAX_STEPS = int(np.ceil(4.5 * H_nom))
    STUCK_WIN = int(np.ceil(0.7 * MAX_STEPS))  # info only

    for e in envs:
        e.max_steps = MAX_STEPS

    if ep % 50 == 0 or ep in (0, int(CURRICULUM_FRAC*EPISODES)):
        print(f"[**dyn**] ep={ep} | STEP_M={STEP_M:.2f} m | DELTA_PSI_DEG=±{max_delta_deg:.3f}° | "
              f"H_nom={H_nom} | MAX_STEPS={MAX_STEPS} | STUCK_WINDOW={STUCK_WIN}\n")

H_big   = max(1, int(np.ceil(L_xy / STEP_M_BIG)))
H_small = max(1, int(np.ceil(L_xy / STEP_M_SMALL)))
TOTAL_STEPS_BUDGET = int(EPISODES * (CURRICULUM_FRAC*3*H_big + (1-CURRICULUM_FRAC)*3*H_small))
EPSILON_DECAY_STEPS = max(20000, int(0.5 * TOTAL_STEPS_BUDGET))
START_EPSILON = 1.0
END_EPSILON   = 0.08
WARMUP_STEPS  = int(0.15 * TOTAL_STEPS_BUDGET)
print(f"[cfg] L_xy={L_xy:.1f} m | eps decay over {EPSILON_DECAY_STEPS} steps (of ~{TOTAL_STEPS_BUDGET}), warmup={WARMUP_STEPS}\n")

def epsilon_by_step(step):
    if step < WARMUP_STEPS:
        return 1.0
    frac = min(1.0, (step - WARMUP_STEPS) / max(1, EPSILON_DECAY_STEPS))
    return START_EPSILON + (END_EPSILON - START_EPSILON) * frac

huber_elementwise = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

@tf.function(autograph=False, jit_compile=False, reduce_retracing=True, experimental_relax_shapes=True)
def train_step(bs, ba, br, bs2, bd, bmask):
    q_next_online = q_net(bs2, training=False)
    minus_inf = tf.cast(-1e9, q_next_online.dtype)
    q_next_online = tf.where(bmask, q_next_online, minus_inf)
    next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)

    q_next_target = target_net(bs2, training=False)
    q_next = tf.gather(q_next_target, next_actions, axis=1, batch_dims=1)

    targets = br + (1.0 - bd) * GAMMA * q_next

    with tf.GradientTape() as tape:
        q_pred_all = q_net(bs, training=True)
        q_pred = tf.gather(q_pred_all, ba[:, None], axis=1, batch_dims=1)
        losses = huber_elementwise(targets, tf.squeeze(q_pred, axis=1))
        loss = tf.reduce_mean(losses)
    grads = tape.gradient(loss, q_net.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 10.0)
    q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss

def _densify_xy_from_idx(path_idx_seq, ds=EXPORT_DS_M):
    xs, ys, ij = [], [], []
    if not path_idx_seq or len(path_idx_seq) < 2:
        return np.array(xs), np.array(ys), ij
    for (i0, j0, k0), (i1, j1, k1) in zip(path_idx_seq[:-1], path_idx_seq[1:]):
        x0, y0 = i0 * DX, j0 * DY
        x1, y1 = i1 * DX, j1 * DY
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(L / max(ds, 1e-6))))
        for t in range(n):
            a = t / float(n)
            xt = x0 + a * (x1 - x0)
            yt = y0 + a * (y1 - y0)
            xs.append(xt); ys.append(yt)
            ii, jj = meters_to_ij(xt, yt)
            ij.append((ii, jj))
    il, jl, kl = path_idx_seq[-1]
    xs.append(il * DX); ys.append(jl * DY); ij.append((il, jl))
    return np.array(xs), np.array(ys), ij

def _min_turn_radius_from_xy(xs, ys):
    if len(xs) < 3:
        return float('inf'), 0.0
    def circumR(x0, y0, x1, y1, x2, y2):
        a = math.hypot(x1 - x0, y1 - y0)
        b = math.hypot(x2 - x1, y2 - y1)
        c = math.hypot(x2 - x0, y2 - y0)
        area2 = abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
        if area2 < 1e-9: return float('inf')
        return (a * b * c) / (2.0 * area2)
    minR = float('inf')
    sum_abs_turn_deg = 0.0
    for i in range(len(xs) - 2):
        R = circumR(xs[i], ys[i], xs[i+1], ys[i+1], xs[i+2], ys[i+2])
        if R < minR: minR = R
        v1 = np.array([xs[i+1] - xs[i], ys[i+1] - ys[i]])
        v2 = np.array([xs[i+2] - xs[i+1], ys[i+2] - ys[i+1]])
        if np.linalg.norm(v1) > 1e-9 and np.linalg.norm(v2) > 1e-9:
            a1 = math.atan2(v1[1], v1[0])
            a2 = math.atan2(v2[1], v2[0])
            d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
            sum_abs_turn_deg += abs(math.degrees(d))
    return minR, sum_abs_turn_deg

def compute_kpis(path_idx_seq):
    ds_kpi = min(DX, DY)
    xs, ys, ij = _densify_xy_from_idx(path_idx_seq, ds=ds_kpi)
    if len(xs) == 0:
        return None

    L = 0.0
    for (i1, j1, k1), (i2, j2, k2) in zip(path_idx_seq[:-1], path_idx_seq[1:]):
        x1, y1, z1 = i1 * DX, j1 * DY, SEABED_Z[i1, j1]
        x2, y2, z2 = i2 * DX, j2 * DY, SEABED_Z[i2, j2]
        L += math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    clear_list = []
    for (ii, jj) in ij:
        d = float(RISK_EDGE_DIST[ii, jj])
        clear_list.append(max(0.0, d))
    min_clear = float(min(clear_list)) if clear_list else float('inf')

    viol_len = 0.0
    for ((i0, j0), (i1, j1)) in zip(ij[:-1], ij[1:]):
        d0 = max(0.0, float(RISK_EDGE_DIST[i0, j0]))
        d1 = max(0.0, float(RISK_EDGE_DIST[i1, j1]))
        segL = math.hypot((i1 - i0) * DX, (j1 - j0) * DY)
        if (d0 < RISK_CLEARANCE_M) or (d1 < RISK_CLEARANCE_M):
            if (d0 - RISK_CLEARANCE_M) * (d1 - RISK_CLEARANCE_M) < 0:
                t = abs(d0 - RISK_CLEARANCE_M) / max(abs(d1 - d0), 1e-9)
                viol_len += min(t, 1.0) * segL
            else:
                viol_len += segL

    slope_vals = [float(SLOPE_DEG[ii, jj]) for (ii, jj) in ij]
    max_slope = float(max(slope_vals)) if slope_vals else 0.0
    mean_slope = float(np.mean(slope_vals)) if slope_vals else 0.0

    total_span = 0.0
    max_span_run = 0.0
    run = 0.0
    for ((i0, j0), (i1, j1)) in zip(ij[:-1], ij[1:]):
        gap = max(0.0, (0.5 * (SEABED_Z[i0, j0] + SEABED_Z[i1, j1]) - SEABED_Z[(i0 + i1)//2, (j0 + j1)//2]
                        if abs(i1 - i0) <= 1 and abs(j1 - j0) <= 1 else 0.0))
        segL = math.hypot((i1 - i0) * DX, (j1 - j0) * DY)
        if gap > gap_thresh:
            run += segL
            total_span += segL
            max_span_run = max(max_span_run, run)
        else:
            run = 0.0

    minR, sum_turn_deg = _min_turn_radius_from_xy(xs, ys)

    return {
        'length_m': L,
        'min_clear_m': min_clear,
        'viol_clear_len_m': viol_len,
        'max_slope_deg': max_slope,
        'mean_slope_deg': mean_slope,
        'total_freespan_m': total_span,
        'max_freespan_run_m': max_span_run,
        'min_turn_radius_m': float(minR),
        'sum_abs_turn_deg': float(sum_turn_deg)
    }


#%% Training loop, KPI Top-3, export, final viz

print("Training starts.")

envs = [SubseaRouteEnv(INVALID_MASK, OBSU, START_IDX, GOAL_IDX, 600)
        for _ in range(N_ENVS)]

global_step = 0
grad_steps = 0
best_return = -1e9
best_path = None

reached_eps = []  # store goal-reaching episodes

set_episode_dynamics(0, envs)

for ep in range(EPISODES):
    set_episode_dynamics(ep, envs)
    for e in envs:
        e.current_episode = ep

    states = [env.reset() for env in envs]
    ep_returns = [0.0] * N_ENVS
    dones = [False] * N_ENVS

    ep_breakdown = {"r_length": 0.0, "r_progress": 0.0, "r_bend": 0.0, "r_obsu": 0.0,
                    "r_span": 0.0, "r_clear": 0.0, "r_slope": 0.0, "r_goal": 0.0}
    eps_sum = 0.0
    eps_count = 0
    reached_goal = False

    while not all(dones):
        global_step += 1
        eps = epsilon_by_step(global_step)
        eps_sum += eps; eps_count += 1

        active_idx = [i for i, d in enumerate(dones) if not d]
        if len(active_idx) == 0: break
        state_batch = np.stack([states[i] for i in active_idx], axis=0)

        q_batch = q_net(state_batch, training=False).numpy()
        valid_masks_batch = np.stack([envs[i].valid_action_mask() for i in active_idx], axis=0)
        q_batch[~valid_masks_batch] = -1e9

        a_batch = np.empty(len(active_idx), dtype=np.int32)
        for t, i_env in enumerate(active_idx):
            if np.random.rand() < eps:
                valids = np.where(valid_masks_batch[t])[0]
                a_batch[t] = int(np.random.choice(valids)) if len(valids) > 0 else int(np.random.randint(NUM_ACTIONS))
            else:
                a_batch[t] = int(np.argmax(q_batch[t]))

        for t, i_env in enumerate(active_idx):
            env = envs[i_env]
            s = states[i_env]
            a = a_batch[t]
            s2, r, done2, info = env.step(a)
            ep_returns[i_env] += r

            if i_env == 0 and "components" in info:
                for k in ep_breakdown.keys():
                    ep_breakdown[k] += float(info["components"].get(k, 0.0))
                if info["components"].get("r_goal", 0.0) > 0.0:
                    reached_goal = True

            next_mask = env.valid_action_mask()
            memory.push(s, a, r, s2, float(done2), next_mask)
            states[i_env] = s2
            dones[i_env] = done2

        if len(memory) >= BATCH_SIZE:
            bs, ba, br, bs2, bd, bmask = memory.sample(BATCH_SIZE)
            bs    = tf.convert_to_tensor(bs, dtype=tf.float32)
            ba    = tf.convert_to_tensor(ba, dtype=tf.int32)
            br    = tf.convert_to_tensor(br, dtype=tf.float32)
            bs2   = tf.convert_to_tensor(bs2, dtype=tf.float32)
            bd    = tf.convert_to_tensor(bd, dtype=tf.float32)
            bmask = tf.convert_to_tensor(bmask, dtype=tf.bool)
            _loss = train_step(bs, ba, br, bs2, bd, bmask)
            grad_steps += 1
            if grad_steps % TARGET_UPDATE_STEPS == 0:
                target_net.set_weights(q_net.get_weights())

    if (ep + 1) % 10 == 0:
        plot_episode_xy(envs[0].path, episode=ep+1, save_dir="episode_frames", show=False, with_contour=False,
                        ep_return=ep_returns[0], ep_eps=(eps_sum / max(1, eps_count)))
        print(f"Episode {ep+1}/{EPISODES} | Return[0]={ep_returns[0]:.1f} | Reached={reached_goal}\n")

    if reached_goal:
        reached_eps.append({
            "ep": ep + 1,
            "return": float(ep_returns[0]),
            "path": list(envs[0].path),
            "eps_avg": float(eps_sum / max(1, eps_count)),
            "breakdown": deepcopy(ep_breakdown),
            "reached": True
        })

print("Training done.")

def compute_kpi_score(length_m, min_clear_m, viol_clear_len_m,
                      max_slope_deg, mean_slope_deg,
                      total_freespan_m, max_freespan_run_m,
                      min_turn_radius_m, sum_abs_turn_deg):
    w_safety = 0.5
    w_construct = 0.3
    w_cost = 0.2

    score_clear  = np.tanh(min_clear_m / 100.0) - np.tanh(viol_clear_len_m / 100.0)
    score_span   = np.exp(-total_freespan_m / 500.0) * np.exp(-max_freespan_run_m / 100.0)
    score_turn   = np.exp(-abs(sum_abs_turn_deg) / 1000.0) * np.tanh(min_turn_radius_m / 500.0)
    score_slope  = np.exp(-max_slope_deg / 30.0) * np.exp(-mean_slope_deg / 15.0)
    score_length = np.exp(-length_m / 5000.0)

    safety = (score_clear + score_span + score_turn + score_slope) / 4.0
    construct = (score_slope + score_turn) / 2.0
    cost = score_length

    kpi_score = w_safety * safety + w_construct * construct + w_cost * cost
    return float(np.clip(kpi_score, 0.0, 1.0))

def _save_kpi_top3_pngs_and_log(reached_list, frames_dir="episode_frames", log_path="kpi_episodes_log.txt"):
    if not reached_list:
        print("[KPI] No goal-reaching episodes."); return None

    ranked = []
    for entry in reached_list:
        kpi = compute_kpis(entry["path"])
        if kpi is None: continue
        entry_k = deepcopy(entry)
        entry_k["kpi"] = kpi
        ranked.append(entry_k)

    if not ranked:
        print("[KPI] No routes to compute KPIs."); return None

    def sort_key(e):
        k = e["kpi"]
        return (k['viol_clear_len_m'], -k['min_clear_m'], k['max_slope_deg'], k['total_freespan_m'],
                k['length_m'], k['sum_abs_turn_deg'])

    ranked.sort(key=sort_key)
    top3 = ranked[:3]

    os.makedirs(frames_dir, exist_ok=True)
    for rank, entry in enumerate(top3, start=1):
        ep_no = entry["ep"]
        path = entry["path"]
        ep_ret = entry["return"]
        ep_eps = entry["eps_avg"]
        plot_episode_xy(path, episode=ep_no, save_dir=frames_dir, show=False, with_contour=False,
                        ep_return=ep_ret, ep_eps=ep_eps)
        src = os.path.join(frames_dir, f"ep_{ep_no:04d}.png")
        dst = os.path.join(frames_dir, f"KPI_Best{rank}_ep_{ep_no:04d}.png")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"[viz] saved {dst}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# KPI-based Top Episodes (sorted by safety/quality, then efficiency)\n\n")
        f.write("# Columns: ep, return, length_m, min_clear_m, viol_clear_len_m, max_slope_deg, mean_slope_deg, "
                "total_freespan_m, max_freespan_run_m, min_turn_radius_m, sum_abs_turn_deg\n")
        for rank, entry in enumerate(top3, start=1):
            k = entry["kpi"]
            length_m            = k["length_m"]
            min_clear_m         = k["min_clear_m"]
            viol_clear_len_m    = k["viol_clear_len_m"]
            max_slope_deg       = k["max_slope_deg"]
            mean_slope_deg      = k["mean_slope_deg"]
            total_freespan_m    = k["total_freespan_m"]
            max_freespan_run_m  = k["max_freespan_run_m"]
            min_turn_radius_m   = k["min_turn_radius_m"]
            sum_abs_turn_deg    = k["sum_abs_turn_deg"]

            kpi_score = compute_kpi_score(length_m, min_clear_m, viol_clear_len_m,
                                          max_slope_deg, mean_slope_deg,
                                          total_freespan_m, max_freespan_run_m,
                                          min_turn_radius_m, sum_abs_turn_deg)
            f.write(
                f"Rank {rank} | ep={entry['ep']} | RL_return={entry['return']:.2f} | KPI_score={kpi_score:.3f} | "
                f"L={k['length_m']:.1f} m | minClr={k['min_clear_m']:.1f} m | violClrLen={k['viol_clear_len_m']:.1f} m | "
                f"maxSlope={k['max_slope_deg']:.2f}° | meanSlope={k['mean_slope_deg']:.2f}° | "
                f"spanTot={k['total_freespan_m']:.1f} m | spanMaxRun={k['max_freespan_run_m']:.1f} m | "
                f"minR={k['min_turn_radius_m']:.1f} m | sumTurn={k['sum_abs_turn_deg']:.1f}°\n"
            )
    print(f"[KPI] saved {log_path}")
    return top3

kpi_top3 = _save_kpi_top3_pngs_and_log(reached_eps, frames_dir="episode_frames", log_path="kpi_episodes_log.txt")
if kpi_top3:
    best_path = list(kpi_top3[0]["path"])
    save_best_path_xyz_dense(best_path, "best_path_xyz.txt", ds=EXPORT_DS_M)
else:
    print("Cannot produce KPI-based Top-3.")

# Final 3D visualization (optional)
SHOW_FINAL_PATH = True
SMOOTH_FINAL_PATH = True

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def visualize_final_path(path_idx_seq, smooth=SMOOTH_FINAL_PATH):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rx, ry, rz = np.where(RISK_MASK)
    if rx.size > 0:
        ax.scatter(rx * DX, ry * DY, rz * DZ, s=1, alpha=0.35, color='black', label='Risk Zones')

    ix, iy = np.where(invalid_xy)
    iz = seabed_k[ix, iy]
    if ix.size > 0:
        ax.scatter(ix * DX, iy * DY, iz * DZ, s=4, alpha=0.4, c='orange', label='OBSU invalid')

    xs = np.arange(0, NX, 4) * DX
    ys = np.arange(0, NY, 4) * DY
    Xs, Ys = np.meshgrid(xs, ys, indexing='ij')
    Zs = SEABED_Z[::4, ::4]
    ax.plot_wireframe(Xs, Ys, Zs, rstride=1, cstride=1, linewidth=0.4, alpha=0.7)

    if smooth:
        xy = [(i * DX, j * DY) for (i, j, k) in path_idx_seq]
        xy_s = chaikin_smooth_xy(xy, n_iter=2)
        xs, ys = xy_s[:, 0], xy_s[:, 1]
        zs = [SEABED_Z[min(max(int(round(x / DX)), 0), NX - 1),
                       min(max(int(round(y / DY)), 0), NY - 1)] for x, y in zip(xs, ys)]
    else:
        xs, ys, zs = [], [], []
        for (i, j, k) in path_idx_seq:
            x, y = i * DX, j * DY
            z = SEABED_Z[i, j]
            xs.append(x); ys.append(y); zs.append(z)

    ax.plot(xs, ys, zs, linewidth=2.0, label='Best Path (KPI)')

    si, sj, _ = START_IDX
    gi, gj, _ = GOAL_IDX
    ax.scatter([si * DX], [sj * DY], [SEABED_Z[si, sj]], s=80, marker='o', c='green', label='Start')
    ax.scatter([gi * DX], [gj * DY], [SEABED_Z[gi, gj]], s=80, marker='^', c='red', label='Goal')

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Depth Z [m]')
    ax.set_title('Final Route (KPI #1)')
    ax.invert_zaxis()
    ax.legend(loc='upper left')
    plt.show()

if SHOW_FINAL_PATH and kpi_top3:
    best_path = list(kpi_top3[0]["path"])
    visualize_final_path(best_path)
