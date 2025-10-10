#!/usr/bin/env python3
"""
Script pour générer scenario_noisy.txt à partir des 400 premières poses
avec les loop closures définies
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Lire vertices_with_loops.txt
data = []
with open('scenario/vertices_with_loops.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            pose_id = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            # Loop IDs
            loop_ids_str = parts[8] if len(parts) > 8 else ''
            loop_ids = []
            if loop_ids_str:
                loop_ids = [int(x) for x in loop_ids_str.split(';')]

            data.append({
                'pose_id': pose_id,
                'tx': tx,
                'ty': ty,
                'tz': tz,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'qw': qw,
                'loop_ids': loop_ids
            })

print(f"Nombre total de poses chargées: {len(data)}")

# Limiter aux 400 premières poses
max_poses = 400
data_subset = data[:max_poses]
print(f"Poses conservées: {len(data_subset)}")

# Filtrer les loop closures pour ne garder que ceux dans [0-400)
for d in data_subset:
    d['loop_ids'] = [lid for lid in d['loop_ids'] if lid < max_poses]

# Compter les loop closures
loop_count = sum(len(d['loop_ids']) for d in data_subset) // 2  # Divisé par 2 car bidirectionnel
print(f"Loop closures dans les 400 premières poses: {loop_count}")

# Sauvegarder scenario_noisy.txt
output_file = 'scenario/scenario_noisy.txt'
with open(output_file, 'w') as f:
    f.write("pose_id,tx,ty,tz,qx,qy,qz,qw,loop_ids\n")

    for d in data_subset:
        loop_ids_str = ';'.join(map(str, sorted(d['loop_ids']))) if d['loop_ids'] else ''
        f.write(f"{d['pose_id']},{d['tx']:.9f},{d['ty']:.9f},{d['tz']:.9f},"
                f"{d['qx']:.9f},{d['qy']:.9f},{d['qz']:.9f},{d['qw']:.9f},{loop_ids_str}\n")

print(f"\nFichier sauvegardé: {output_file}")
print(f"Poses avec loop closures: {sum(1 for d in data_subset if d['loop_ids'])}")
