#!/usr/bin/env python3
"""
Module pour convertir les fichiers de scénario en format g2o
"""
import numpy as np
from scipy.spatial.transform import Rotation

def load_scenario(filename):
    """Charge un fichier de scénario et retourne les données structurées"""
    data = []
    with open(filename, 'r') as f:
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
    return data

def compute_relative_transform(pose1, pose2):
    """
    Calcule la transformation relative entre deux poses SE(3)
    T_relative = inv(T1) * T2
    """
    # Pose 1
    R1 = Rotation.from_quat([pose1['qx'], pose1['qy'], pose1['qz'], pose1['qw']]).as_matrix()
    t1 = np.array([pose1['tx'], pose1['ty'], pose1['tz']])

    # Pose 2
    R2 = Rotation.from_quat([pose2['qx'], pose2['qy'], pose2['qz'], pose2['qw']]).as_matrix()
    t2 = np.array([pose2['tx'], pose2['ty'], pose2['tz']])

    # Transformation relative
    R_rel = R1.T @ R2
    t_rel = R1.T @ (t2 - t1)

    # Convertir en quaternion
    r_rel = Rotation.from_matrix(R_rel)
    q_rel = r_rel.as_quat()  # [qx, qy, qz, qw]

    return t_rel[0], t_rel[1], t_rel[2], q_rel[0], q_rel[1], q_rel[2], q_rel[3]

def write_g2o_from_scenario(scenario_file, output_file, odom_info=500.0, loop_info=700.0, ref_file_for_loops=None, loop_sampling=1):
    """
    Convertit un fichier de scénario en format g2o

    Args:
        scenario_file: Fichier d'entrée (scenario_noisy.txt ou scenario_ref.txt)
        output_file: Fichier g2o de sortie
        odom_info: Valeur de la matrice d'information pour l'odométrie
        loop_info: Valeur de la matrice d'information pour les loop closures
        ref_file_for_loops: Fichier de référence pour calculer les loop closures (optionnel)
                           Si fourni, les loops seront calculés depuis les poses de référence
        loop_sampling: Prendre 1 loop sur X (default: 1 = tous). 0 = pas de loops
    """
    print(f"Lecture de {scenario_file}...")
    data = load_scenario(scenario_file)
    print(f"  {len(data)} poses chargées")

    # Si on doit calculer les loops depuis un fichier de référence
    data_ref_for_loops = None
    if ref_file_for_loops:
        print(f"  Chargement des poses de référence depuis {ref_file_for_loops} pour les loop closures...")
        data_ref_for_loops = load_scenario(ref_file_for_loops)
        print(f"    {len(data_ref_for_loops)} poses de référence chargées")

    # Construire les paires de loop closures
    loop_pairs = []
    if loop_sampling > 0:
        for i, d in enumerate(data):
            for loop_id in d['loop_ids']:
                if i < loop_id and loop_id < len(data):  # Éviter doublons et indices invalides
                    loop_pairs.append((i, loop_id))

        # Appliquer le sampling
        if loop_sampling > 1:
            loop_pairs_original = len(loop_pairs)
            loop_pairs = loop_pairs[::loop_sampling]  # Prendre 1 sur loop_sampling
            print(f"  {loop_pairs_original} loop closures détectées, {len(loop_pairs)} après sampling (1/{loop_sampling})")
        else:
            print(f"  {len(loop_pairs)} loop closures détectées")
    else:
        print(f"  0 loop closures (sampling=0)")

    # Écrire le fichier g2o
    print(f"Écriture de {output_file}...")
    with open(output_file, 'w') as f:
        # Écrire les vertices
        for i, d in enumerate(data):
            f.write(f"VERTEX_SE3:QUAT {i} {d['tx']:.9f} {d['ty']:.9f} {d['tz']:.9f} "
                   f"{d['qx']:.9f} {d['qy']:.9f} {d['qz']:.9f} {d['qw']:.9f}\n")

        # Fixer le premier vertex
        f.write(f"FIX 0\n")

        # Écrire les edges d'odométrie (séquentielles)
        for i in range(len(data) - 1):
            tx, ty, tz, qx, qy, qz, qw = compute_relative_transform(data[i], data[i+1])

            # Matrice d'information 6x6 diagonale (upper triangular storage)
            # Format: I11 I12 I13 I14 I15 I16 I22 I23 I24 I25 I26 I33 I34 I35 I36 I44 I45 I46 I55 I56 I66
            info_vals = [odom_info, 0, 0, 0, 0, 0,  # ligne 1
                        odom_info, 0, 0, 0, 0,      # ligne 2
                        odom_info, 0, 0, 0,         # ligne 3
                        odom_info, 0, 0,            # ligne 4
                        odom_info, 0,               # ligne 5
                        odom_info]                   # ligne 6

            info_str = ' '.join(f"{v:.1f}" for v in info_vals)

            f.write(f"EDGE_SE3:QUAT {i} {i+1} {tx:.9f} {ty:.9f} {tz:.9f} "
                   f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} {info_str}\n")

        # Écrire les edges de loop closures
        for i, j in loop_pairs:
            # Si on a un fichier de référence, utiliser les poses de référence pour les loops
            if data_ref_for_loops:
                tx, ty, tz, qx, qy, qz, qw = compute_relative_transform(data_ref_for_loops[i], data_ref_for_loops[j])
            else:
                tx, ty, tz, qx, qy, qz, qw = compute_relative_transform(data[i], data[j])

            # Matrice d'information pour loops (plus élevée)
            info_vals = [loop_info, 0, 0, 0, 0, 0,
                        loop_info, 0, 0, 0, 0,
                        loop_info, 0, 0, 0,
                        loop_info, 0, 0,
                        loop_info, 0,
                        loop_info]

            info_str = ' '.join(f"{v:.1f}" for v in info_vals)

            f.write(f"EDGE_SE3:QUAT {i} {j} {tx:.9f} {ty:.9f} {tz:.9f} "
                   f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f} {info_str}\n")

    print(f"Fichier g2o créé avec succès!")
    print(f"  Vertices: {len(data)}")
    print(f"  Odometry edges: {len(data) - 1}")
    print(f"  Loop closure edges: {len(loop_pairs)}")
    print(f"  Total edges: {len(data) - 1 + len(loop_pairs)}")

def read_g2o_optimized(filename):
    """
    Lit les poses optimisées depuis un fichier g2o

    Returns:
        Liste de dictionnaires avec les poses optimisées
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('VERTEX_SE3:QUAT'):
                parts = line.strip().split()
                pose_id = int(parts[1])
                tx, ty, tz = float(parts[2]), float(parts[3]), float(parts[4])
                qx, qy, qz, qw = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])

                data.append({
                    'pose_id': pose_id,
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                    'qx': qx,
                    'qy': qy,
                    'qz': qz,
                    'qw': qw
                })

    return data

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convertir un scénario en format g2o')
    parser.add_argument('--input', type=str, default='scenario/scenario_noisy.txt',
                       help='Fichier de scénario en entrée')
    parser.add_argument('--output', type=str, default='scenario_noisy.g2o',
                       help='Fichier g2o en sortie')
    parser.add_argument('--odom-info', type=float, default=500.0,
                       help='Information matrix value pour odométrie')
    parser.add_argument('--loop-info', type=float, default=700.0,
                       help='Information matrix value pour loop closures')
    args = parser.parse_args()

    write_g2o_from_scenario(args.input, args.output, args.odom_info, args.loop_info)
