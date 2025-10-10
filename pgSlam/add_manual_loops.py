#!/usr/bin/env python3
"""
Script pour ajouter des loop closures manuelles entre deux segments de trajectoire
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Lire le fichier poses_with_loops.txt
data = []
with open('scenario/poses_with_loops.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip header

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 9:
            kf_idx = int(parts[0])
            img_idx = int(parts[1])
            tx_c2w, ty_c2w, tz_c2w = float(parts[2]), float(parts[3]), float(parts[4])
            qx_c2w, qy_c2w, qz_c2w, qw_c2w = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])

            # Inverser la transformation: camera-to-world -> world-to-camera
            r_c2w = Rotation.from_quat([qx_c2w, qy_c2w, qz_c2w, qw_c2w])
            R_c2w = r_c2w.as_matrix()
            t_c2w = np.array([tx_c2w, ty_c2w, tz_c2w])

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            r_w2c = Rotation.from_matrix(R_w2c)
            q_w2c = r_w2c.as_quat()

            tx, ty, tz = t_w2c[0], t_w2c[1], t_w2c[2]
            qx, qy, qz, qw = q_w2c[0], q_w2c[1], q_w2c[2], q_w2c[3]

            # Loop IDs existants
            loop_ids_str = parts[9] if len(parts) > 9 else ''
            loop_ids = []
            if loop_ids_str:
                loop_ids = [int(x) for x in loop_ids_str.split(';')]

            data.append({
                'kf_idx': kf_idx,
                'img_idx': img_idx,
                'pos': (tx, ty, tz),
                'quat': (qx, qy, qz, qw),
                'loop_ids': loop_ids.copy()
            })

print(f"Nombre de poses chargées: {len(data)}")

# Définir les segments
segment1_start = 0
segment1_end = 52
segment2_start = 99
segment2_end = 140

# Vérifier les limites
segment2_end = min(segment2_end, len(data))

print(f"\nSegment 1: poses {segment1_start} à {segment1_end} (kf_idx {data[segment1_start]['kf_idx']} à {data[segment1_end]['kf_idx']})")
print(f"Segment 2: poses {segment2_start} à {segment2_end-1} (kf_idx {data[segment2_start]['kf_idx']} à {data[segment2_end-1]['kf_idx']})")

# Trouver les loops basés sur la proximité spatiale
new_loops = []
distance_threshold = 3.0  # Distance maximale en mètres

for i in range(segment1_start, segment1_end + 1):
    pos1 = np.array(data[i]['pos'])

    # Trouver la pose la plus proche dans le segment 2
    min_dist = float('inf')
    closest_j = -1

    for j in range(segment2_start, segment2_end):
        pos2 = np.array(data[j]['pos'])
        dist = np.linalg.norm(pos2 - pos1)

        if dist < min_dist and dist < distance_threshold:
            min_dist = dist
            closest_j = j

    if closest_j != -1:
        new_loops.append((i, closest_j, min_dist))
        # Ajouter les loop closures mutuellement
        if closest_j not in data[i]['loop_ids']:
            data[i]['loop_ids'].append(closest_j)
        if i not in data[closest_j]['loop_ids']:
            data[closest_j]['loop_ids'].append(i)

print(f"\nNombre de nouveaux loops trouvés: {len(new_loops)}")
print("\nNouvelles loop closures:")
for i, (idx1, idx2, dist) in enumerate(new_loops[:10]):  # Afficher les 10 premiers
    kf1 = data[idx1]['kf_idx']
    kf2 = data[idx2]['kf_idx']
    print(f"  {i+1}. list_idx {idx1} (kf {kf1}) <-> list_idx {idx2} (kf {kf2}), distance: {dist:.2f}m")
if len(new_loops) > 10:
    print(f"  ... et {len(new_loops) - 10} loops supplémentaires")

# Sauvegarder le fichier mis à jour
output_file = 'scenario/poses_with_manual_loops.txt'
with open(output_file, 'w') as f:
    # Header
    f.write("kf_idx,img_idx,tx,ty,tz,qx,qy,qz,qw,loop_ids\n")

    # Écrire toutes les poses avec les loops mis à jour
    for d in data:
        # Retourner à camera-to-world pour l'écriture
        tx, ty, tz = d['pos']
        qx, qy, qz, qw = d['quat']

        # Inverser w2c -> c2w
        r_w2c = Rotation.from_quat([qx, qy, qz, qw])
        R_w2c = r_w2c.as_matrix()
        t_w2c = np.array([tx, ty, tz])

        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        r_c2w = Rotation.from_matrix(R_c2w)
        q_c2w = r_c2w.as_quat()

        tx_c2w, ty_c2w, tz_c2w = t_c2w[0], t_c2w[1], t_c2w[2]
        qx_c2w, qy_c2w, qz_c2w, qw_c2w = q_c2w[0], q_c2w[1], q_c2w[2], q_c2w[3]

        # Loop IDs
        loop_ids_str = ';'.join(map(str, sorted(d['loop_ids']))) if d['loop_ids'] else ''

        f.write(f"{d['kf_idx']},{d['img_idx']},{tx_c2w:.9f},{ty_c2w:.9f},{tz_c2w:.9f},"
               f"{qx_c2w:.9f},{qy_c2w:.9f},{qz_c2w:.9f},{qw_c2w:.9f},{loop_ids_str}\n")

print(f"\nFichier sauvegardé: {output_file}")
print(f"Total de poses avec loop closures: {sum(1 for d in data if d['loop_ids'])}")
