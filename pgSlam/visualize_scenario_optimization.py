#!/usr/bin/env python3
"""
Script pour visualiser le résultat de l'optimisation:
- Scenario noisy (avant optimisation)
- Scenario optimized (après optimisation)
- Scenario ref (ground truth)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from g2o_io_scenario import load_scenario, read_g2o_optimized

def load_optimized_txt(filename):
    """Charge un fichier de poses optimisées au format texte"""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                pose_id = int(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

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

def visualize_optimization_results(noisy_file='scenario/scenario_noisy.txt',
                                   optimized_file='scenario_optimized.txt',
                                   ref_file='scenario/scenario_ref.txt'):
    """
    Visualise les trois scénarios côte à côte
    """
    print("Chargement des scénarios...")

    # Charger les données
    data_noisy = load_scenario(noisy_file)
    print(f"  Noisy: {len(data_noisy)} poses")

    data_optimized = load_optimized_txt(optimized_file)
    print(f"  Optimized: {len(data_optimized)} poses")

    data_ref = load_scenario(ref_file)
    print(f"  Reference: {len(data_ref)} poses")

    # Extraire les positions
    X_noisy = [d['tx'] for d in data_noisy]
    Y_noisy = [d['ty'] for d in data_noisy]
    Z_noisy = [d['tz'] for d in data_noisy]

    X_opt = [d['tx'] for d in data_optimized]
    Y_opt = [d['ty'] for d in data_optimized]
    Z_opt = [d['tz'] for d in data_optimized]

    X_ref = [d['tx'] for d in data_ref]
    Y_ref = [d['ty'] for d in data_ref]
    Z_ref = [d['tz'] for d in data_ref]

    # Construire les loop pairs depuis noisy
    loop_pairs = []
    for i, d in enumerate(data_noisy):
        for loop_id in d['loop_ids']:
            if i < loop_id:
                loop_pairs.append((i, loop_id))

    print(f"  Loop closures: {len(loop_pairs)}")

    # Segments
    seg1_start, seg1_end = 0, 88
    seg2_start, seg2_end = 304, min(392, len(X_noisy)-1)

    # Créer la figure avec 4 subplots (2x2)
    fig = plt.figure(figsize=(20, 16))

    # Subplot 1: Noisy
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(X_noisy, Y_noisy, Z_noisy, 'gray', linewidth=1.5, alpha=0.4, label='Trajectoire')
    ax1.plot(X_noisy[seg1_start:seg1_end+1], Y_noisy[seg1_start:seg1_end+1], Z_noisy[seg1_start:seg1_end+1],
            'g-', linewidth=2.5, alpha=0.8, label=f'Seg 1 [{seg1_start}-{seg1_end}]')
    ax1.plot(X_noisy[seg2_start:seg2_end+1], Y_noisy[seg2_start:seg2_end+1], Z_noisy[seg2_start:seg2_end+1],
            'orange', linewidth=2.5, alpha=0.8, label=f'Seg 2 [{seg2_start}-{seg2_end}]')

    # Loop closures
    for idx, (i, j) in enumerate(loop_pairs[::5]):  # Afficher 1 sur 5 pour lisibilité
        ax1.plot([X_noisy[i], X_noisy[j]], [Y_noisy[i], Y_noisy[j]], [Z_noisy[i], Z_noisy[j]],
                'r-', linewidth=1.5, alpha=0.5, label='Loops' if idx == 0 else '')

    ax1.scatter(X_noisy[0], Y_noisy[0], Z_noisy[0], c='green', s=150, marker='o',
              edgecolors='black', linewidths=2, label='Start')
    ax1.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    ax1.set_title(f'NOISY (Avant optimisation)\n{len(data_noisy)} poses, {len(loop_pairs)} loops',
                 fontsize=13, fontweight='bold', color='red')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)

    # Subplot 2: Optimized
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(X_opt, Y_opt, Z_opt, 'gray', linewidth=1.5, alpha=0.4, label='Trajectoire')
    ax2.plot(X_opt[seg1_start:seg1_end+1], Y_opt[seg1_start:seg1_end+1], Z_opt[seg1_start:seg1_end+1],
            'g-', linewidth=2.5, alpha=0.8, label=f'Seg 1 [{seg1_start}-{seg1_end}]')
    ax2.plot(X_opt[seg2_start:seg2_end+1], Y_opt[seg2_start:seg2_end+1], Z_opt[seg2_start:seg2_end+1],
            'orange', linewidth=2.5, alpha=0.8, label=f'Seg 2 [{seg2_start}-{seg2_end}]')

    # Loop closures
    for idx, (i, j) in enumerate(loop_pairs[::5]):
        ax2.plot([X_opt[i], X_opt[j]], [Y_opt[i], Y_opt[j]], [Z_opt[i], Z_opt[j]],
                'r-', linewidth=1.5, alpha=0.5, label='Loops' if idx == 0 else '')

    ax2.scatter(X_opt[0], Y_opt[0], Z_opt[0], c='green', s=150, marker='o',
              edgecolors='black', linewidths=2, label='Start')
    ax2.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax2.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    ax2.set_title(f'OPTIMIZED (Après g2o)\n{len(data_optimized)} poses',
                 fontsize=13, fontweight='bold', color='blue')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)

    # Subplot 3: Reference (Ground Truth)
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(X_ref, Y_ref, Z_ref, 'gray', linewidth=1.5, alpha=0.4, label='Trajectoire')
    ax3.plot(X_ref[seg1_start:seg1_end+1], Y_ref[seg1_start:seg1_end+1], Z_ref[seg1_start:seg1_end+1],
            'g-', linewidth=2.5, alpha=0.8, label=f'Seg 1 [{seg1_start}-{seg1_end}]')
    ax3.plot(X_ref[seg2_start:seg2_end+1], Y_ref[seg2_start:seg2_end+1], Z_ref[seg2_start:seg2_end+1],
            'orange', linewidth=2.5, alpha=0.8, label=f'Seg 2 [{seg2_start}-{seg2_end}] (=Seg1)')

    # Loop closures (distance ~0)
    for idx, (i, j) in enumerate(loop_pairs[::5]):
        ax3.plot([X_ref[i], X_ref[j]], [Y_ref[i], Y_ref[j]], [Z_ref[i], Z_ref[j]],
                'r-', linewidth=1.5, alpha=0.5, label='Loops (dist≈0)' if idx == 0 else '')

    ax3.scatter(X_ref[0], Y_ref[0], Z_ref[0], c='green', s=150, marker='o',
              edgecolors='black', linewidths=2, label='Start')
    ax3.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    ax3.set_title(f'REFERENCE (Ground Truth)\n{len(data_ref)} poses, segments alignés',
                 fontsize=13, fontweight='bold', color='green')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.view_init(elev=20, azim=45)

    # Subplot 4: Comparaison des trois trajectoires (sans loops)
    ax4 = fig.add_subplot(224, projection='3d')

    # Tracer les trois trajectoires avec des couleurs distinctes
    # Reference (ground truth) en vert
    ax4.plot(X_ref, Y_ref, Z_ref, 'green', linewidth=2.5, alpha=0.7, label='Reference (GT)')

    # Noisy en rouge
    ax4.plot(X_noisy, Y_noisy, Z_noisy, 'red', linewidth=2, alpha=0.6, label='Noisy (avant optim)')

    # Optimized en bleu
    ax4.plot(X_opt, Y_opt, Z_opt, 'blue', linewidth=2, alpha=0.8, label='Optimized (après g2o)')

    # Marquer les segments pour référence
    ax4.scatter(X_ref[seg1_start], Y_ref[seg1_start], Z_ref[seg1_start],
               c='green', s=200, marker='o', edgecolors='black', linewidths=2, label='Start')
    ax4.scatter(X_ref[seg2_start], Y_ref[seg2_start], Z_ref[seg2_start],
               c='orange', s=150, marker='s', edgecolors='black', linewidths=2, label=f'Seg2 start ({seg2_start})')

    ax4.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax4.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    ax4.set_title(f'COMPARAISON: Ref (vert) / Noisy (rouge) / Optimized (bleu)\n(Sans loops - voir l\'amélioration)',
                 fontsize=13, fontweight='bold', color='purple')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

    print("\nVisualisation complète!")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualiser les résultats de l\'optimisation')
    parser.add_argument('--noisy', type=str, default='scenario/scenario_noisy.txt',
                       help='Fichier scenario noisy')
    parser.add_argument('--optimized', type=str, default='scenario_optimized.txt',
                       help='Fichier scenario optimized')
    parser.add_argument('--ref', type=str, default='scenario/scenario_ref.txt',
                       help='Fichier scenario reference')
    args = parser.parse_args()

    visualize_optimization_results(args.noisy, args.optimized, args.ref)
