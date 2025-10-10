#!/usr/bin/env python3
"""
Script pour ajouter des loop closures entre deux segments spécifiques
"""
import numpy as np
from scipy.spatial.transform import Rotation

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

            # Inverser: camera-to-world -> world-to-camera
            r_c2w = Rotation.from_quat([qx_c2w, qy_c2w, qz_c2w, qw_c2w])
            R_c2w = r_c2w.as_matrix()
            t_c2w = np.array([tx_c2w, ty_c2w, tz_c2w])

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            tx, ty, tz = t_w2c[0], t_w2c[1], t_w2c[2]

            data.append({
                'pose_id': pose_id,
                'pos': (tx, ty, tz),
                'loop_ids': []
            })

print(f"Nombre total de poses: {len(data)}")

# Définir les segments
seg1_start = 0
seg1_end = 88
seg2_start = 304
seg2_end = 304 + 88  # 392

print(f"\nSegment 1: poses {seg1_start} à {seg1_end}")
print(f"Segment 2: poses {seg2_start} à {seg2_end}")

# Créer des loop closures 1-pour-1 entre les segments
new_loops = []

# Mapping direct: pose i <-> pose (seg2_start + i)
for i in range(seg1_start, min(seg1_end + 1, len(data))):
    offset = i - seg1_start
    j = seg2_start + offset

    # Vérifier que j est dans les limites
    if j < len(data) and j <= seg2_end:
        pos1 = np.array(data[i]['pos'])
        pos2 = np.array(data[j]['pos'])
        dist = np.linalg.norm(pos2 - pos1)

        new_loops.append((i, j, dist))
        # Ajouter les loop closures mutuellement
        data[i]['loop_ids'].append(j)
        data[j]['loop_ids'].append(i)

print(f"\nNombre de loop closures créées: {len(new_loops)}")
print("\nPremières loop closures:")
for idx, (i, j, dist) in enumerate(new_loops[:10]):
    print(f"  {idx+1}. Pose {i} (ID {data[i]['pose_id']}) <-> Pose {j} (ID {data[j]['pose_id']}), distance: {dist:.2f}m")
if len(new_loops) > 10:
    print(f"  ... et {len(new_loops) - 10} loops supplémentaires")

# Sauvegarder les données avec loops
output_file = 'scenario/vertices_with_loops.txt'
with open(output_file, 'w') as f:
    f.write("pose_id,tx,ty,tz,qx,qy,qz,qw,loop_ids\n")

    for d in data:
        # Position en w2c pour écriture
        tx, ty, tz = d['pos']

        # Loop IDs
        loop_ids_str = ';'.join(map(str, sorted(d['loop_ids']))) if d['loop_ids'] else ''

        f.write(f"{d['pose_id']},{tx:.9f},{ty:.9f},{tz:.9f},0,0,0,1,{loop_ids_str}\n")

print(f"\nFichier sauvegardé: {output_file}")
print(f"Poses avec loop closures: {sum(1 for d in data if d['loop_ids'])}")
