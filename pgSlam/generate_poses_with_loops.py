#!/usr/bin/env python3
"""
Script pour générer un fichier de poses avec loop closures
À partir des fichiers du dossier scenario/
"""
import ast

# Lire id_kf_2_idx_img.txt (liste Python d'IDs)
with open('scenario/id_kf_2_idx_img.txt', 'r') as f:
    kf_ids = ast.literal_eval(f.read().strip())

print(f"Nombre de keyframes: {len(kf_ids)}")

# Créer un mapping: img_idx -> kf_idx (position dans la liste)
img_to_kf_idx = {}
for kf_idx, img_idx in enumerate(kf_ids):
    img_name = f"img{img_idx:05d}"
    img_to_kf_idx[img_name] = kf_idx

print(f"Mapping créé: {len(img_to_kf_idx)} images")

# Lire vertices_stan.txt (poses)
poses = {}
with open('scenario/vertices_stan.txt', 'r') as f:
    lines = f.readlines()

header = lines[0].strip()
print(f"Header: {header}")

for line in lines[1:]:
    parts = line.strip().split(',')
    if len(parts) == 8:
        pose_id = int(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        poses[pose_id] = (tx, ty, tz, qx, qy, qz, qw)

print(f"Nombre de poses lues: {len(poses)}")

# Lire loop_all.txt (loop closures)
loops = []
with open('scenario/loop_all.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            img1 = parts[0]
            img2 = parts[1]

            # Vérifier si les deux images sont dans nos keyframes
            if img1 in img_to_kf_idx and img2 in img_to_kf_idx:
                kf_idx1 = img_to_kf_idx[img1]
                kf_idx2 = img_to_kf_idx[img2]
                loops.append((kf_idx1, kf_idx2))

print(f"Nombre de loop closures: {len(loops)}")

# Créer un dictionnaire: kf_idx -> liste des loop partners
loop_dict = {}
for kf_idx1, kf_idx2 in loops:
    if kf_idx1 not in loop_dict:
        loop_dict[kf_idx1] = []
    if kf_idx2 not in loop_dict:
        loop_dict[kf_idx2] = []
    loop_dict[kf_idx1].append(kf_idx2)
    loop_dict[kf_idx2].append(kf_idx1)

print(f"Keyframes avec loop closures: {len(loop_dict)}")

# Générer le fichier de sortie
output_file = 'scenario/poses_with_loops.txt'
with open(output_file, 'w') as f:
    # Header
    f.write("kf_idx,img_idx,tx,ty,tz,qx,qy,qz,qw,loop_ids\n")

    # Pour chaque keyframe
    for kf_idx, img_idx in enumerate(kf_ids):
        # Récupérer la pose correspondante
        if img_idx in poses:
            tx, ty, tz, qx, qy, qz, qw = poses[img_idx]

            # Récupérer les loop closures
            if kf_idx in loop_dict:
                loop_ids_str = ';'.join(map(str, sorted(loop_dict[kf_idx])))
            else:
                loop_ids_str = ''

            # Écrire la ligne
            f.write(f"{kf_idx},{img_idx},{tx:.9f},{ty:.9f},{tz:.9f},"
                   f"{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},{loop_ids_str}\n")
        else:
            print(f"Warning: Pose non trouvée pour img_idx {img_idx}")

print(f"\nFichier généré: {output_file}")
print(f"Format: kf_idx,img_idx,tx,ty,tz,qx,qy,qz,qw,loop_ids")
print(f"  - kf_idx: Index du keyframe (position dans id_kf_2_idx_img)")
print(f"  - img_idx: ID de l'image originale")
print(f"  - tx,ty,tz,qx,qy,qz,qw: Pose (translation + quaternion)")
print(f"  - loop_ids: Liste des kf_idx en loop closure (séparés par ';')")

# Afficher quelques exemples
print(f"\nPremières lignes du fichier:")
with open(output_file, 'r') as f:
    for i, line in enumerate(f):
        print(f"  {line.strip()}")
        if i >= 5:
            break

# Statistiques
num_with_loops = sum(1 for kf_idx in range(len(kf_ids)) if kf_idx in loop_dict)
print(f"\nStatistiques:")
print(f"  Total keyframes: {len(kf_ids)}")
print(f"  Keyframes avec loop closures: {num_with_loops}")
print(f"  Keyframes sans loop closures: {len(kf_ids) - num_with_loops}")
print(f"  Total loop closures: {len(loops)}")
