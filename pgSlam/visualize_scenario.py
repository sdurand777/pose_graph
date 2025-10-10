#!/usr/bin/env python3
"""
Script pour visualiser les poses du scénario avec loop closures
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import argparse

# Parser les arguments
parser = argparse.ArgumentParser(description='Visualiser les poses du scénario')
parser.add_argument('--num-poses', type=int, default=None,
                    help='Nombre de poses à afficher (défaut: toutes)')
args = parser.parse_args()

# Lire le fichier poses_with_loops.txt
data = []
kf_idx_to_list_idx = {}  # Mapping kf_idx -> position dans la liste data

with open('scenario/poses_with_loops.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for list_idx, line in enumerate(lines):
        parts = line.strip().split(',')
        if len(parts) >= 9:
            kf_idx = int(parts[0])
            img_idx = int(parts[1])
            tx_c2w, ty_c2w, tz_c2w = float(parts[2]), float(parts[3]), float(parts[4])
            qx_c2w, qy_c2w, qz_c2w, qw_c2w = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])

            # Inverser la transformation: camera-to-world -> world-to-camera
            # T_c2w = [R_c2w | t_c2w]
            # T_w2c = [R_c2w^T | -R_c2w^T * t_c2w]

            # Créer la matrice de rotation depuis le quaternion
            r_c2w = Rotation.from_quat([qx_c2w, qy_c2w, qz_c2w, qw_c2w])
            R_c2w = r_c2w.as_matrix()
            t_c2w = np.array([tx_c2w, ty_c2w, tz_c2w])

            # Inverser
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            # Convertir en quaternion
            r_w2c = Rotation.from_matrix(R_w2c)
            q_w2c = r_w2c.as_quat()  # [qx, qy, qz, qw]

            tx, ty, tz = t_w2c[0], t_w2c[1], t_w2c[2]
            qx, qy, qz, qw = q_w2c[0], q_w2c[1], q_w2c[2], q_w2c[3]

            # Loop IDs
            loop_ids_str = parts[9] if len(parts) > 9 else ''
            loop_ids = []
            if loop_ids_str:
                loop_ids = [int(x) for x in loop_ids_str.split(';')]

            data.append({
                'kf_idx': kf_idx,
                'img_idx': img_idx,
                'pos': (tx, ty, tz),
                'quat': (qx, qy, qz, qw),
                'loop_ids': loop_ids
            })

            # Créer le mapping
            kf_idx_to_list_idx[kf_idx] = list_idx

# Construire les paires de loop closures avec les bons indices
loop_pairs = []
for list_idx, d in enumerate(data):
    kf_idx = d['kf_idx']
    for loop_id in d['loop_ids']:
        # Vérifier que les deux kf_idx existent dans notre liste
        if kf_idx in kf_idx_to_list_idx and loop_id in kf_idx_to_list_idx:
            idx1 = kf_idx_to_list_idx[kf_idx]
            idx2 = kf_idx_to_list_idx[loop_id]
            # Éviter les doublons
            if idx1 < idx2:
                loop_pairs.append((idx1, idx2))

print(f"Nombre de poses: {len(data)}")
print(f"Nombre de loop closures: {len(loop_pairs)}")

# Extraire les positions
X = [d['pos'][0] for d in data]
Y = [d['pos'][1] for d in data]
Z = [d['pos'][2] for d in data]

# Statistiques
print(f"\nRange de positions:")
print(f"  X: [{min(X):.2f}, {max(X):.2f}] m")
print(f"  Y: [{min(Y):.2f}, {max(Y):.2f}] m")
print(f"  Z: [{min(Z):.2f}, {max(Z):.2f}] m")

# Distance totale parcourue
total_dist = 0
for i in range(1, len(data)):
    dx = X[i] - X[i-1]
    dy = Y[i] - Y[i-1]
    dz = Z[i] - Z[i-1]
    total_dist += np.sqrt(dx**2 + dy**2 + dz**2)

print(f"\nDistance totale: {total_dist:.2f} m")

# Visualisation 1: Trajectoire avec IDs des poses
max_kf_to_show = len(data) if args.num_poses is None else min(args.num_poses, len(data))
print(f"Affichage de {max_kf_to_show} poses")
data_subset = data[:max_kf_to_show]
X_subset = X[:max_kf_to_show]
Y_subset = Y[:max_kf_to_show]
Z_subset = Z[:max_kf_to_show]

fig1 = plt.figure(figsize=(14, 10))
ax_ids = fig1.add_subplot(111, projection='3d')

# Tracer la trajectoire
ax_ids.plot(X_subset, Y_subset, Z_subset, 'b-', linewidth=2, alpha=0.7, label='Trajectoire')

# Afficher toutes les poses avec leurs IDs
step = max(1, len(X_subset) // 40)  # Afficher environ 40 labels
for i in range(0, len(X_subset), step):
    kf_idx = data_subset[i]['kf_idx']
    ax_ids.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='blue', s=50, alpha=0.8)
    ax_ids.text(X_subset[i], Y_subset[i], Z_subset[i], f'  {kf_idx}', fontsize=8, alpha=0.9)

# Start et End
ax_ids.scatter(X_subset[0], Y_subset[0], Z_subset[0], c='green', s=200, marker='o', label='Start', edgecolors='black', linewidths=2)
ax_ids.text(X_subset[0], Y_subset[0], Z_subset[0], f'  START({data_subset[0]["kf_idx"]})', fontsize=10, fontweight='bold', color='green')

ax_ids.scatter(X_subset[-1], Y_subset[-1], Z_subset[-1], c='red', s=200, marker='s', label='End (kf {})'.format(data_subset[-1]["kf_idx"]), edgecolors='black', linewidths=2)
ax_ids.text(X_subset[-1], Y_subset[-1], Z_subset[-1], f'  END({data_subset[-1]["kf_idx"]})', fontsize=10, fontweight='bold', color='red')

# Loop closures en rouge épais (seulement celles dans les 200 premières)
loop_pairs_subset = [(i, j) for (i, j) in loop_pairs if i < max_kf_to_show and j < max_kf_to_show]
for (i, j) in loop_pairs_subset:
    kf1 = data[i]['kf_idx']
    kf2 = data[j]['kf_idx']
    ax_ids.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                'r-', linewidth=3, alpha=0.8, label='Loop Closure' if (i, j) == loop_pairs_subset[0] else '')
    # Afficher les IDs des loop closures
    mid_x = (X[i] + X[j]) / 2
    mid_y = (Y[i] + Y[j]) / 2
    mid_z = (Z[i] + Z[j]) / 2
    ax_ids.text(mid_x, mid_y, mid_z, f'{kf1}↔{kf2}', fontsize=9, color='red', fontweight='bold')

ax_ids.set_xlabel('X (m)', fontsize=12)
ax_ids.set_ylabel('Y (m)', fontsize=12)
ax_ids.set_zlabel('Z (m)', fontsize=12)
ax_ids.set_title(f'Trajectoire 3D avec IDs des poses\n({len(data_subset)} poses affichées, {len(loop_pairs_subset)} loop closures)',
                 fontsize=14, fontweight='bold')
ax_ids.legend(loc='best', fontsize=10)
ax_ids.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistiques sur les loop closures
if loop_pairs:
    loop_distances = []
    loop_time_gaps = []

    for (i, j) in loop_pairs:
        # Distance spatiale
        dx = X[i] - X[j]
        dy = Y[i] - Y[j]
        dz = Z[i] - Z[j]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        loop_distances.append(dist)

        # Écart temporel (en nombre de poses)
        time_gap = abs(j - i)
        loop_time_gaps.append(time_gap)

    print(f"\nStatistiques des loop closures:")
    print(f"  Distance spatiale moyenne: {np.mean(loop_distances):.2f} m")
    print(f"  Distance spatiale min/max: [{min(loop_distances):.2f}, {max(loop_distances):.2f}] m")
    print(f"  Écart temporel moyen: {np.mean(loop_time_gaps):.0f} poses")
    print(f"  Écart temporel min/max: [{min(loop_time_gaps)}, {max(loop_time_gaps)}] poses")

    # Afficher les 5 premières loop closures
    print(f"\nPremières loop closures:")
    for i, (idx1, idx2) in enumerate(loop_pairs[:5]):
        kf1 = data[idx1]['kf_idx']
        kf2 = data[idx2]['kf_idx']
        print(f"  {i+1}. Pose {kf1} <-> Pose {kf2} "
              f"(list_idx: {idx1}<->{idx2}, "
              f"distance: {loop_distances[i]:.2f}m)")

print("\nTerminé!")
