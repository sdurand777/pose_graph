#!/usr/bin/env python3
"""
Script pour générer scenario_ref.txt où les poses [304-392] sont remplacées
par des copies exactes des poses [0-88] (ground truth pour loop closures)
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Lire scenario_noisy.txt
data = []
with open('scenario/scenario_noisy.txt', 'r') as f:
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

print(f"Nombre de poses dans scenario_noisy.txt: {len(data)}")

# Remplacer les poses [304-392] par des copies de [0-88]
seg1_start = 0
seg1_end = 88
seg2_start = 304
seg2_end = 392

replacements = 0
for i in range(seg1_start, min(seg1_end + 1, len(data))):
    offset = i - seg1_start
    j = seg2_start + offset

    if j < len(data) and j <= seg2_end:
        # Copier la pose i vers la pose j (sauf pose_id et loop_ids)
        data[j]['tx'] = data[i]['tx']
        data[j]['ty'] = data[i]['ty']
        data[j]['tz'] = data[i]['tz']
        data[j]['qx'] = data[i]['qx']
        data[j]['qy'] = data[i]['qy']
        data[j]['qz'] = data[i]['qz']
        data[j]['qw'] = data[i]['qw']
        replacements += 1

print(f"Nombre de poses remplacées: {replacements}")
print(f"Segment 1 [{seg1_start}-{seg1_end}] copié vers Segment 2 [{seg2_start}-{min(seg2_start + replacements - 1, seg2_end)}]")

# Sauvegarder scenario_ref.txt
output_file = 'scenario/scenario_ref.txt'
with open(output_file, 'w') as f:
    f.write("pose_id,tx,ty,tz,qx,qy,qz,qw,loop_ids\n")

    for d in data:
        loop_ids_str = ';'.join(map(str, sorted(d['loop_ids']))) if d['loop_ids'] else ''
        f.write(f"{d['pose_id']},{d['tx']:.9f},{d['ty']:.9f},{d['tz']:.9f},"
                f"{d['qx']:.9f},{d['qy']:.9f},{d['qz']:.9f},{d['qw']:.9f},{loop_ids_str}\n")

print(f"\nFichier sauvegardé: {output_file}")

# Vérification: calculer la distance entre quelques paires
print("\nVérification des distances entre paires (doivent être ~0 dans scenario_ref):")
for i in range(seg1_start, min(seg1_start + 5, seg1_end + 1)):
    j = seg2_start + (i - seg1_start)
    if j < len(data):
        pos1 = np.array([data[i]['tx'], data[i]['ty'], data[i]['tz']])
        pos2 = np.array([data[j]['tx'], data[j]['ty'], data[j]['tz']])
        dist = np.linalg.norm(pos2 - pos1)
        print(f"  Pose {i} <-> Pose {j}: distance = {dist:.6f}m")
