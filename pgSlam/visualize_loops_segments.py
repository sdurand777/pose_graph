#!/usr/bin/env python3
"""
Script pour visualiser les poses avec loop closures entre segments
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Parser les arguments
parser = argparse.ArgumentParser(description='Visualiser les poses avec loops entre segments')
parser.add_argument('--num-poses', type=int, default=None,
                    help='Nombre de poses à afficher (défaut: toutes)')
args = parser.parse_args()

# Lire vertices_with_loops.txt
data = []
with open('scenario/vertices_with_loops.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            pose_id = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])

            # Loop IDs
            loop_ids_str = parts[8] if len(parts) > 8 else ''
            loop_ids = []
            if loop_ids_str:
                loop_ids = [int(x) for x in loop_ids_str.split(';')]

            data.append({
                'pose_id': pose_id,
                'pos': (tx, ty, tz),
                'loop_ids': loop_ids
            })

print(f"Nombre total de poses: {len(data)}")

# Construire les paires de loop closures
loop_pairs = []
for i, d in enumerate(data):
    for loop_id in d['loop_ids']:
        if i < loop_id:  # Éviter les doublons
            loop_pairs.append((i, loop_id))

print(f"Nombre de loop closures: {len(loop_pairs)}")

# Extraire les positions
X = [d['pos'][0] for d in data]
Y = [d['pos'][1] for d in data]
Z = [d['pos'][2] for d in data]

# Déterminer le nombre de poses à afficher
max_to_show = len(data) if args.num_poses is None else min(args.num_poses, len(data))
print(f"Affichage de {max_to_show} poses")

data_subset = data[:max_to_show]
X_subset = X[:max_to_show]
Y_subset = Y[:max_to_show]
Z_subset = Z[:max_to_show]

# Visualisation 3D
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection='3d')

# Tracer la trajectoire complète
ax.plot(X_subset, Y_subset, Z_subset, 'gray', linewidth=1.5, alpha=0.5, label='Trajectoire')

# Segment 1: [0-88] en vert
seg1_start = 0
seg1_end = min(88, len(X_subset))
ax.plot(X_subset[seg1_start:seg1_end+1], Y_subset[seg1_start:seg1_end+1], Z_subset[seg1_start:seg1_end+1],
        'g-', linewidth=3, alpha=0.8, label=f'Segment 1 [{seg1_start}-{seg1_end}]')
for i in range(seg1_start, seg1_end+1, max(1, (seg1_end - seg1_start) // 10)):
    ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='green', s=80, alpha=0.8, edgecolors='darkgreen', linewidths=1)

# Segment 2: [304-392] en orange
seg2_start = 304
seg2_end = min(392, len(X_subset))
if seg2_start < len(X_subset):
    ax.plot(X_subset[seg2_start:seg2_end+1], Y_subset[seg2_start:seg2_end+1], Z_subset[seg2_start:seg2_end+1],
            'orange', linewidth=3, alpha=0.8, label=f'Segment 2 [{seg2_start}-{seg2_end}]')
    for i in range(seg2_start, min(seg2_end+1, len(X_subset)), max(1, (seg2_end - seg2_start) // 10)):
        ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='orange', s=80, alpha=0.8, edgecolors='darkorange', linewidths=1)

# Afficher quelques IDs de poses
step = max(1, len(X_subset) // 30)
for i in range(0, len(X_subset), step):
    pose_id = data_subset[i]['pose_id']
    ax.text(X_subset[i], Y_subset[i], Z_subset[i], f' {i}', fontsize=7, alpha=0.7)

# Start et End
ax.scatter(X_subset[0], Y_subset[0], Z_subset[0], c='green', s=300, marker='o',
          label='Start', edgecolors='black', linewidths=3)
ax.text(X_subset[0], Y_subset[0], Z_subset[0], f'  START(0)',
        fontsize=12, fontweight='bold', color='green')

if len(X_subset) > 1:
    ax.scatter(X_subset[-1], Y_subset[-1], Z_subset[-1], c='red', s=300, marker='s',
              label=f'End ({len(X_subset)-1})', edgecolors='black', linewidths=3)

# Loop closures en rouge épais
loop_pairs_subset = [(i, j) for (i, j) in loop_pairs if i < max_to_show and j < max_to_show]
print(f"\nLoop closures affichées: {len(loop_pairs_subset)}")

loop_count = 0
for (i, j) in loop_pairs_subset:
    dist = np.linalg.norm(np.array(data[i]['pos']) - np.array(data[j]['pos']))

    ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
            'r-', linewidth=2.5, alpha=0.7, label='Loop Closures' if loop_count == 0 else '')

    # Afficher quelques labels
    if loop_count < 8:  # Afficher les 8 premiers
        mid_x = (X[i] + X[j]) / 2
        mid_y = (Y[i] + Y[j]) / 2
        mid_z = (Z[i] + Z[j]) / 2
        ax.text(mid_x, mid_y, mid_z, f'{i}↔{j}\n{dist:.1f}m',
                fontsize=7, color='red', fontweight='bold', alpha=0.8)

    loop_count += 1

ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
ax.set_title(f'Trajectoire 3D avec Loop Closures entre Segments\n' +
             f'Vert: Segment 1 [0-88], Orange: Segment 2 [304-392]\n' +
             f'({len(data_subset)} poses, {len(loop_pairs_subset)} loops)',
             fontsize=15, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)

# Ajuster la vue pour mieux voir
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# Statistiques détaillées
if loop_pairs_subset:
    distances = [np.linalg.norm(np.array(data[i]['pos']) - np.array(data[j]['pos']))
                 for (i, j) in loop_pairs_subset]
    print(f"\nStatistiques des loop closures:")
    print(f"  Distance moyenne: {np.mean(distances):.2f}m")
    print(f"  Distance min: {min(distances):.2f}m")
    print(f"  Distance max: {max(distances):.2f}m")

    print(f"\nDétails des loop closures:")
    for idx, (i, j) in enumerate(loop_pairs_subset[:15]):
        dist = np.linalg.norm(np.array(data[i]['pos']) - np.array(data[j]['pos']))
        print(f"  {idx+1}. Pose {i} <-> Pose {j}, distance: {dist:.2f}m")
    if len(loop_pairs_subset) > 15:
        print(f"  ... et {len(loop_pairs_subset) - 15} loops supplémentaires")

print("\nTerminé!")
