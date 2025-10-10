#!/usr/bin/env python3
"""
Script pour visualiser les poses avec les loop closures manuelles
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import argparse

# Parser les arguments
parser = argparse.ArgumentParser(description='Visualiser les poses avec loop closures manuelles')
parser.add_argument('--num-poses', type=int, default=None,
                    help='Nombre de poses à afficher (défaut: toutes)')
args = parser.parse_args()

# Lire le fichier avec les loops manuels
data = []
kf_idx_to_list_idx = {}

with open('scenario/poses_with_manual_loops.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for list_idx, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) >= 9:
            kf_idx = int(parts[0])
            img_idx = int(parts[1])
            tx_c2w, ty_c2w, tz_c2w = float(parts[2]), float(parts[3]), float(parts[4])
            qx_c2w, qy_c2w, qz_c2w, qw_c2w = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])

            # Inverser la transformation
            r_c2w = Rotation.from_quat([qx_c2w, qy_c2w, qz_c2w, qw_c2w])
            R_c2w = r_c2w.as_matrix()
            t_c2w = np.array([tx_c2w, ty_c2w, tz_c2w])

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            r_w2c = Rotation.from_matrix(R_w2c)
            q_w2c = r_w2c.as_quat()

            tx, ty, tz = t_w2c[0], t_w2c[1], t_w2c[2]

            # Loop IDs
            loop_ids_str = parts[9] if len(parts) > 9 else ''
            loop_ids = []
            if loop_ids_str:
                loop_ids = [int(x) for x in loop_ids_str.split(';')]

            data.append({
                'kf_idx': kf_idx,
                'img_idx': img_idx,
                'pos': (tx, ty, tz),
                'loop_ids': loop_ids
            })
            kf_idx_to_list_idx[kf_idx] = list_idx

# Construire les paires de loop closures
loop_pairs = []
for list_idx, d in enumerate(data):
    kf_idx = d['kf_idx']
    for loop_id in d['loop_ids']:
        if kf_idx in kf_idx_to_list_idx and loop_id in kf_idx_to_list_idx:
            idx1 = kf_idx_to_list_idx[kf_idx]
            idx2 = kf_idx_to_list_idx[loop_id]
            if idx1 < idx2:
                loop_pairs.append((idx1, idx2))

print(f"Nombre de poses: {len(data)}")
print(f"Nombre de loop closures: {len(loop_pairs)}")

# Extraire les positions
X = [d['pos'][0] for d in data]
Y = [d['pos'][1] for d in data]
Z = [d['pos'][2] for d in data]

# Visualisation des N premières poses
max_kf_to_show = len(data) if args.num_poses is None else min(args.num_poses, len(data))
print(f"Affichage de {max_kf_to_show} poses")
data_subset = data[:max_kf_to_show]
X_subset = X[:max_kf_to_show]
Y_subset = Y[:max_kf_to_show]
Z_subset = Z[:max_kf_to_show]

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Tracer la trajectoire
ax.plot(X_subset, Y_subset, Z_subset, 'b-', linewidth=2, alpha=0.7, label='Trajectoire')

# Afficher les poses avec leurs IDs
step = max(1, len(X_subset) // 30)
for i in range(0, len(X_subset), step):
    kf_idx = data_subset[i]['kf_idx']
    ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='blue', s=50, alpha=0.8)
    ax.text(X_subset[i], Y_subset[i], Z_subset[i], f'  {kf_idx}', fontsize=8, alpha=0.9)

# Marquer les segments
# Segment 1: [0-52]
for i in range(0, min(53, len(X_subset))):
    ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='green', s=30, alpha=0.6)

# Segment 2: [99-140]
for i in range(99, min(140, len(X_subset))):
    ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='orange', s=30, alpha=0.6)

# Start et End
ax.scatter(X_subset[0], Y_subset[0], Z_subset[0], c='green', s=200, marker='o',
          label='Start', edgecolors='black', linewidths=2)
ax.text(X_subset[0], Y_subset[0], Z_subset[0], f'  START({data_subset[0]["kf_idx"]})',
        fontsize=10, fontweight='bold', color='green')

# Loop closures
loop_pairs_subset = [(i, j) for (i, j) in loop_pairs if i < max_kf_to_show and j < max_kf_to_show]
print(f"\nLoop closures dans les 140 premières poses: {len(loop_pairs_subset)}")

for idx, (i, j) in enumerate(loop_pairs_subset):
    kf1 = data[i]['kf_idx']
    kf2 = data[j]['kf_idx']

    # Calculer la distance
    dist = np.linalg.norm(np.array(data[i]['pos']) - np.array(data[j]['pos']))

    ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
            'r-', linewidth=2.5, alpha=0.8, label='Loop Closure' if idx == 0 else '')

    # Afficher quelques labels
    if idx < 5 or (i >= 0 and i <= 52 and j >= 99):  # Afficher les loops manuels
        mid_x = (X[i] + X[j]) / 2
        mid_y = (Y[i] + Y[j]) / 2
        mid_z = (Z[i] + Z[j]) / 2
        ax.text(mid_x, mid_y, mid_z, f'{kf1}↔{kf2}\n{dist:.1f}m',
                fontsize=7, color='red', fontweight='bold')

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title(f'Trajectoire 3D avec Loop Closures Manuelles\n' +
             f'Vert: Segment 1 [0-52], Orange: Segment 2 [99-140]\n' +
             f'({len(data_subset)} poses, {len(loop_pairs_subset)} loops)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Afficher les statistiques des loops
print("\nLoop closures entre segments [0-52] et [99-140]:")
manual_loops = [(i, j) for (i, j) in loop_pairs_subset if i <= 52 and j >= 99]
print(f"Nombre: {len(manual_loops)}")
for idx, (i, j) in enumerate(manual_loops[:15]):
    kf1 = data[i]['kf_idx']
    kf2 = data[j]['kf_idx']
    dist = np.linalg.norm(np.array(data[i]['pos']) - np.array(data[j]['pos']))
    print(f"  {idx+1}. kf {kf1} (idx {i}) <-> kf {kf2} (idx {j}), distance: {dist:.2f}m")
if len(manual_loops) > 15:
    print(f"  ... et {len(manual_loops) - 15} loops supplémentaires")
