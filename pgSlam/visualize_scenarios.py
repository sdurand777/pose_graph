#!/usr/bin/env python3
"""
Script pour visualiser et comparer scenario_noisy.txt et scenario_ref.txt
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_scenario(filename):
    """Charge un fichier de scénario"""
    data = []
    with open(filename, 'r') as f:
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
    return data

# Charger les deux scénarios
print("Chargement des scénarios...")
data_noisy = load_scenario('scenario/scenario_noisy.txt')
data_ref = load_scenario('scenario/scenario_ref.txt')

print(f"Scenario noisy: {len(data_noisy)} poses")
print(f"Scenario ref: {len(data_ref)} poses")

# Extraire les positions
X_noisy = [d['pos'][0] for d in data_noisy]
Y_noisy = [d['pos'][1] for d in data_noisy]
Z_noisy = [d['pos'][2] for d in data_noisy]

X_ref = [d['pos'][0] for d in data_ref]
Y_ref = [d['pos'][1] for d in data_ref]
Z_ref = [d['pos'][2] for d in data_ref]

# Construire les paires de loop closures
loop_pairs = []
for i, d in enumerate(data_noisy):
    for loop_id in d['loop_ids']:
        if i < loop_id:  # Éviter les doublons
            loop_pairs.append((i, loop_id))

print(f"Loop closures: {len(loop_pairs)}")

# Calculer les distances pour les loops dans les deux scénarios
print("\nComparaison des distances des loop closures:")
print(f"{'Pair':<15} {'Noisy (m)':<12} {'Ref (m)':<12} {'Diff (m)':<12}")
print("-" * 55)

distances_noisy = []
distances_ref = []

for i, j in loop_pairs[:10]:  # Afficher les 10 premières
    pos1_noisy = np.array(data_noisy[i]['pos'])
    pos2_noisy = np.array(data_noisy[j]['pos'])
    dist_noisy = np.linalg.norm(pos2_noisy - pos1_noisy)
    distances_noisy.append(dist_noisy)

    pos1_ref = np.array(data_ref[i]['pos'])
    pos2_ref = np.array(data_ref[j]['pos'])
    dist_ref = np.linalg.norm(pos2_ref - pos1_ref)
    distances_ref.append(dist_ref)

    diff = dist_noisy - dist_ref
    print(f"{i}↔{j:<12} {dist_noisy:<12.4f} {dist_ref:<12.4f} {diff:<12.4f}")

# Calculer toutes les distances
all_distances_noisy = []
all_distances_ref = []
for i, j in loop_pairs:
    pos1_noisy = np.array(data_noisy[i]['pos'])
    pos2_noisy = np.array(data_noisy[j]['pos'])
    all_distances_noisy.append(np.linalg.norm(pos2_noisy - pos1_noisy))

    pos1_ref = np.array(data_ref[i]['pos'])
    pos2_ref = np.array(data_ref[j]['pos'])
    all_distances_ref.append(np.linalg.norm(pos2_ref - pos1_ref))

print(f"\nStatistiques des distances de loop closures:")
print(f"Noisy - Moyenne: {np.mean(all_distances_noisy):.4f}m, Max: {np.max(all_distances_noisy):.4f}m")
print(f"Ref   - Moyenne: {np.mean(all_distances_ref):.4f}m, Max: {np.max(all_distances_ref):.4f}m")

# Visualisation
fig = plt.figure(figsize=(20, 10))

# Subplot 1: Scenario noisy
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(X_noisy, Y_noisy, Z_noisy, 'gray', linewidth=1.5, alpha=0.5, label='Trajectoire')

# Segment 1: [0-88] en vert
seg1_start, seg1_end = 0, 88
ax1.plot(X_noisy[seg1_start:seg1_end+1], Y_noisy[seg1_start:seg1_end+1], Z_noisy[seg1_start:seg1_end+1],
        'g-', linewidth=3, alpha=0.8, label=f'Segment 1 [{seg1_start}-{seg1_end}]')

# Segment 2: [304-392] en orange
seg2_start, seg2_end = 304, min(392, len(X_noisy)-1)
ax1.plot(X_noisy[seg2_start:seg2_end+1], Y_noisy[seg2_start:seg2_end+1], Z_noisy[seg2_start:seg2_end+1],
        'orange', linewidth=3, alpha=0.8, label=f'Segment 2 [{seg2_start}-{seg2_end}]')

# Loop closures
for idx, (i, j) in enumerate(loop_pairs):
    ax1.plot([X_noisy[i], X_noisy[j]], [Y_noisy[i], Y_noisy[j]], [Z_noisy[i], Z_noisy[j]],
            'r-', linewidth=2, alpha=0.6, label='Loop Closures' if idx == 0 else '')

ax1.scatter(X_noisy[0], Y_noisy[0], Z_noisy[0], c='green', s=200, marker='o',
          edgecolors='black', linewidths=2, label='Start')

ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax1.set_title(f'Scenario NOISY\n({len(data_noisy)} poses, {len(loop_pairs)} loops)',
             fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.view_init(elev=20, azim=45)

# Subplot 2: Scenario ref
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(X_ref, Y_ref, Z_ref, 'gray', linewidth=1.5, alpha=0.5, label='Trajectoire')

# Segment 1: [0-88] en vert
ax2.plot(X_ref[seg1_start:seg1_end+1], Y_ref[seg1_start:seg1_end+1], Z_ref[seg1_start:seg1_end+1],
        'g-', linewidth=3, alpha=0.8, label=f'Segment 1 [{seg1_start}-{seg1_end}]')

# Segment 2: [304-392] en orange (identique à segment 1)
ax2.plot(X_ref[seg2_start:seg2_end+1], Y_ref[seg2_start:seg2_end+1], Z_ref[seg2_start:seg2_end+1],
        'orange', linewidth=3, alpha=0.8, label=f'Segment 2 [{seg2_start}-{seg2_end}] (copie)')

# Loop closures (doivent être de distance ~0)
for idx, (i, j) in enumerate(loop_pairs):
    ax2.plot([X_ref[i], X_ref[j]], [Y_ref[i], Y_ref[j]], [Z_ref[i], Z_ref[j]],
            'r-', linewidth=2, alpha=0.6, label='Loop Closures (dist≈0)' if idx == 0 else '')

ax2.scatter(X_ref[0], Y_ref[0], Z_ref[0], c='green', s=200, marker='o',
          edgecolors='black', linewidths=2, label='Start')

ax2.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax2.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax2.set_title(f'Scenario REF (Ground Truth)\n({len(data_ref)} poses, segments alignés)',
             fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

print("\nVisualization complète!")
