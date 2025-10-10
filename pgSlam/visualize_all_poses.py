#!/usr/bin/env python3
"""
Script pour visualiser toutes les poses de vertices_stan.txt
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import argparse

# Parser les arguments
parser = argparse.ArgumentParser(description='Visualiser toutes les poses')
parser.add_argument('--num-poses', type=int, default=None,
                    help='Nombre de poses à afficher (défaut: toutes)')
args = parser.parse_args()

# Lire vertices_stan.txt
data = []
with open('scenario/vertices_stan.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            pose_id = int(parts[0])
            tx_c2w, ty_c2w, tz_c2w = float(parts[1]), float(parts[2]), float(parts[3])
            qx_c2w, qy_c2w, qz_c2w, qw_c2w = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            # Inverser la transformation: camera-to-world -> world-to-camera
            r_c2w = Rotation.from_quat([qx_c2w, qy_c2w, qz_c2w, qw_c2w])
            R_c2w = r_c2w.as_matrix()
            t_c2w = np.array([tx_c2w, ty_c2w, tz_c2w])

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            tx, ty, tz = t_w2c[0], t_w2c[1], t_w2c[2]

            data.append({
                'pose_id': pose_id,
                'pos': (tx, ty, tz)
            })

print(f"Nombre total de poses chargées: {len(data)}")

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

# Déterminer le nombre de poses à afficher
max_to_show = len(data) if args.num_poses is None else min(args.num_poses, len(data))
print(f"Affichage de {max_to_show} poses")

data_subset = data[:max_to_show]
X_subset = X[:max_to_show]
Y_subset = Y[:max_to_show]
Z_subset = Z[:max_to_show]

# Visualisation 3D
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Tracer la trajectoire
ax.plot(X_subset, Y_subset, Z_subset, 'b-', linewidth=2, alpha=0.7, label='Trajectoire')

# Afficher quelques poses avec leurs IDs
step = max(1, len(X_subset) // 50)  # Afficher environ 50 labels
for i in range(0, len(X_subset), step):
    pose_id = data_subset[i]['pose_id']
    ax.scatter(X_subset[i], Y_subset[i], Z_subset[i], c='blue', s=50, alpha=0.8)
    ax.text(X_subset[i], Y_subset[i], Z_subset[i], f'  {pose_id}', fontsize=8, alpha=0.9)

# Start et End
ax.scatter(X_subset[0], Y_subset[0], Z_subset[0], c='green', s=200, marker='o',
          label='Start', edgecolors='black', linewidths=2)
ax.text(X_subset[0], Y_subset[0], Z_subset[0], f'  START({data_subset[0]["pose_id"]})',
        fontsize=10, fontweight='bold', color='green')

ax.scatter(X_subset[-1], Y_subset[-1], Z_subset[-1], c='red', s=200, marker='s',
          label=f'End (ID {data_subset[-1]["pose_id"]})', edgecolors='black', linewidths=2)
ax.text(X_subset[-1], Y_subset[-1], Z_subset[-1], f'  END({data_subset[-1]["pose_id"]})',
        fontsize=10, fontweight='bold', color='red')

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title(f'Trajectoire 3D - Toutes les poses\n({len(data_subset)} poses affichées sur {len(data)} total)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTerminé!")
