#!/usr/bin/env python3
"""
Script complet pour optimiser le scenario avec visualisation et métriques
Inspiré de main_3d.py
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from g2o_io_scenario import write_g2o_from_scenario, read_g2o_optimized, load_scenario
from compute_scenario_metrics import compute_pose_errors_3d

def visualize_single(data, title, color='blue', loop_pairs=None):
    """Visualise une seule trajectoire avec loop closures optionnels"""
    X = [d['tx'] for d in data]
    Y = [d['ty'] for d in data]
    Z = [d['tz'] for d in data]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(X, Y, Z, color=color, linewidth=2, alpha=0.8, label='Trajectoire')
    ax.scatter(X[0], Y[0], Z[0], c='green', s=200, marker='o',
              edgecolors='black', linewidths=2, label='Start')
    ax.scatter(X[-1], Y[-1], Z[-1], c='red', s=200, marker='s',
              edgecolors='black', linewidths=2, label='End')

    # Afficher les loop closures si fournis
    if loop_pairs:
        for idx, (i, j) in enumerate(loop_pairs):
            if i < len(X) and j < len(X):
                ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]],
                       'cyan', linewidth=1.5, alpha=0.6,
                       label='Loop Closures' if idx == 0 else '')

        # Afficher quelques labels de distance
        for idx, (i, j) in enumerate(loop_pairs[::max(1, len(loop_pairs)//5)]):
            if i < len(X) and j < len(X):
                pos_i = np.array([X[i], Y[i], Z[i]])
                pos_j = np.array([X[j], Y[j], Z[j]])
                dist = np.linalg.norm(pos_j - pos_i)
                mid_x = (X[i] + X[j]) / 2
                mid_y = (Y[i] + Y[j]) / 2
                mid_z = (Z[i] + Z[j]) / 2
                ax.text(mid_x, mid_y, mid_z, f'{i}↔{j}\n{dist:.1f}m',
                       fontsize=7, color='cyan', fontweight='bold', alpha=0.8)

    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def visualize_two(data_ref, data_opt):
    """Compare reference vs optimized (comme drawTwo3D)"""
    X_ref = [d['tx'] for d in data_ref]
    Y_ref = [d['ty'] for d in data_ref]
    Z_ref = [d['tz'] for d in data_ref]

    X_opt = [d['tx'] for d in data_opt]
    Y_opt = [d['ty'] for d in data_opt]
    Z_opt = [d['tz'] for d in data_opt]

    fig = plt.figure(figsize=(18, 8))

    # Reference
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(X_ref, Y_ref, Z_ref, 'green', linewidth=2.5, alpha=0.8, label='Reference (GT)')
    ax1.scatter(X_ref[0], Y_ref[0], Z_ref[0], c='green', s=200, marker='o',
               edgecolors='black', linewidths=2, label='Start')
    ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax1.set_title('REFERENCE (Ground Truth)', fontsize=13, fontweight='bold', color='green')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)

    # Optimized
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(X_opt, Y_opt, Z_opt, 'blue', linewidth=2.5, alpha=0.8, label='Optimized')
    ax2.scatter(X_opt[0], Y_opt[0], Z_opt[0], c='green', s=200, marker='o',
               edgecolors='black', linewidths=2, label='Start')
    ax2.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax2.set_title('OPTIMIZED (Après g2o)', fontsize=13, fontweight='bold', color='blue')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def visualize_three(data_ref, data_opt, data_noisy):
    """Compare les trois trajectoires (comme drawThree3D)"""
    X_ref = [d['tx'] for d in data_ref]
    Y_ref = [d['ty'] for d in data_ref]
    Z_ref = [d['tz'] for d in data_ref]

    X_opt = [d['tx'] for d in data_opt]
    Y_opt = [d['ty'] for d in data_opt]
    Z_opt = [d['tz'] for d in data_opt]

    X_noisy = [d['tx'] for d in data_noisy]
    Y_noisy = [d['ty'] for d in data_noisy]
    Z_noisy = [d['tz'] for d in data_noisy]

    fig = plt.figure(figsize=(20, 6))

    # Reference
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(X_ref, Y_ref, Z_ref, 'green', linewidth=2.5, alpha=0.8, label='Reference')
    ax1.scatter(X_ref[0], Y_ref[0], Z_ref[0], c='green', s=150, marker='o',
               edgecolors='black', linewidths=2)
    ax1.set_xlabel('X (m)', fontweight='bold')
    ax1.set_ylabel('Y (m)', fontweight='bold')
    ax1.set_zlabel('Z (m)', fontweight='bold')
    ax1.set_title('Reference\n(Ground Truth)', fontsize=12, fontweight='bold', color='green')
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=20, azim=45)

    # Optimized
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(X_opt, Y_opt, Z_opt, 'blue', linewidth=2.5, alpha=0.8, label='Optimized')
    ax2.scatter(X_opt[0], Y_opt[0], Z_opt[0], c='green', s=150, marker='o',
               edgecolors='black', linewidths=2)
    ax2.set_xlabel('X (m)', fontweight='bold')
    ax2.set_ylabel('Y (m)', fontweight='bold')
    ax2.set_zlabel('Z (m)', fontweight='bold')
    ax2.set_title('Optimized\n(Après g2o)', fontsize=12, fontweight='bold', color='blue')
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=20, azim=45)

    # Noisy
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(X_noisy, Y_noisy, Z_noisy, 'red', linewidth=2.5, alpha=0.8, label='Noisy')
    ax3.scatter(X_noisy[0], Y_noisy[0], Z_noisy[0], c='green', s=150, marker='o',
               edgecolors='black', linewidths=2)
    ax3.set_xlabel('X (m)', fontweight='bold')
    ax3.set_ylabel('Y (m)', fontweight='bold')
    ax3.set_zlabel('Z (m)', fontweight='bold')
    ax3.set_title('Noisy\n(Avant optim)', fontsize=12, fontweight='bold', color='red')
    ax3.grid(True, alpha=0.3)
    ax3.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def visualize_three_overlayed(data_ref, data_opt, data_noisy):
    """Affiche les trois trajectoires superposées sur le même graphique"""
    X_ref = [d['tx'] for d in data_ref]
    Y_ref = [d['ty'] for d in data_ref]
    Z_ref = [d['tz'] for d in data_ref]

    X_opt = [d['tx'] for d in data_opt]
    Y_opt = [d['ty'] for d in data_opt]
    Z_opt = [d['tz'] for d in data_opt]

    X_noisy = [d['tx'] for d in data_noisy]
    Y_noisy = [d['ty'] for d in data_noisy]
    Z_noisy = [d['tz'] for d in data_noisy]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer les trois trajectoires
    ax.plot(X_ref, Y_ref, Z_ref, 'green', linewidth=3, alpha=0.8, label='Reference (GT)')
    ax.plot(X_noisy, Y_noisy, Z_noisy, 'red', linewidth=2, alpha=0.6, label='Noisy (avant optim)')
    ax.plot(X_opt, Y_opt, Z_opt, 'blue', linewidth=2.5, alpha=0.7, label='Optimized (après g2o)')

    # Marquer le début
    ax.scatter(X_ref[0], Y_ref[0], Z_ref[0], c='green', s=200, marker='o',
              edgecolors='black', linewidths=3, label='Start', zorder=10)

    # Marquer quelques points clés pour voir l'alignement
    seg2_start = 304
    if seg2_start < len(X_ref):
        ax.scatter(X_ref[seg2_start], Y_ref[seg2_start], Z_ref[seg2_start],
                  c='orange', s=150, marker='s', edgecolors='black', linewidths=2,
                  label=f'Segment 2 start (pose {seg2_start})', zorder=10)

    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des 3 Trajectoires (Superposées)\nVert=Reference | Rouge=Noisy | Bleu=Optimized',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def plot_errors(pos_errors_noisy, ori_errors_noisy, pos_errors_opt, ori_errors_opt,
                stats_noisy, stats_opt):
    """Affiche les graphiques d'erreurs (comme plotErrors3D)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    n_poses = len(pos_errors_noisy)
    pose_indices = np.arange(n_poses)

    # Position errors
    ax1.plot(pose_indices, pos_errors_noisy, 'r-', linewidth=1.5, alpha=0.7,
            label=f'Noisy (RMSE: {stats_noisy["position"]["rmse"]:.3f}m)')
    ax1.plot(pose_indices, pos_errors_opt, 'b-', linewidth=1.5, alpha=0.7,
            label=f'Optimized (RMSE: {stats_opt["position"]["rmse"]:.3f}m)')

    ax1.axhline(y=stats_noisy['position']['rmse'], color='red', linestyle='--',
               linewidth=1, alpha=0.5)
    ax1.axhline(y=stats_opt['position']['rmse'], color='blue', linestyle='--',
               linewidth=1, alpha=0.5)

    ax1.set_xlabel('Pose Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Position Error (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Position Error per Pose (vs Reference)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Orientation errors
    ori_errors_noisy_deg = np.rad2deg(ori_errors_noisy)
    ori_errors_opt_deg = np.rad2deg(ori_errors_opt)
    rmse_noisy_deg = np.rad2deg(stats_noisy['orientation']['rmse'])
    rmse_opt_deg = np.rad2deg(stats_opt['orientation']['rmse'])

    ax2.plot(pose_indices, ori_errors_noisy_deg, 'r-', linewidth=1.5, alpha=0.7,
            label=f'Noisy (RMSE: {rmse_noisy_deg:.3f}°)')
    ax2.plot(pose_indices, ori_errors_opt_deg, 'b-', linewidth=1.5, alpha=0.7,
            label=f'Optimized (RMSE: {rmse_opt_deg:.3f}°)')

    ax2.axhline(y=rmse_noisy_deg, color='red', linestyle='--',
               linewidth=1, alpha=0.5)
    ax2.axhline(y=rmse_opt_deg, color='blue', linestyle='--',
               linewidth=1, alpha=0.5)

    ax2.set_xlabel('Pose Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Orientation Error (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title('Orientation Error per Pose (vs Reference)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_metrics(stats_noisy, stats_opt):
    """Affiche les métriques formatées"""
    print("\n" + "="*70)
    print("MÉTRIQUES D'ERREUR")
    print("="*70)

    print("\nPOSITION ERRORS (meters):")
    print(f"{'Metric':<15} {'Noisy':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-"*70)

    for metric in ['rmse', 'mean', 'max']:
        noisy_val = stats_noisy['position'][metric]
        opt_val = stats_opt['position'][metric]
        improvement = ((noisy_val - opt_val) / noisy_val * 100) if noisy_val > 0 else 0

        print(f"{metric.upper():<15} {noisy_val:<15.4f} {opt_val:<15.4f} {improvement:<15.2f}%")

    print("\nORIENTATION ERRORS (degrees):")
    print(f"{'Metric':<15} {'Noisy':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-"*70)

    for metric in ['rmse', 'mean', 'max']:
        noisy_val = np.rad2deg(stats_noisy['orientation'][metric])
        opt_val = np.rad2deg(stats_opt['orientation'][metric])
        improvement = ((noisy_val - opt_val) / noisy_val * 100) if noisy_val > 0 else 0

        print(f"{metric.upper():<15} {noisy_val:<15.4f} {opt_val:<15.4f} {improvement:<15.2f}%")

    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Optimiser et visualiser le scenario')
    parser.add_argument('--input', type=str, default='scenario/scenario_noisy.txt',
                       help='Fichier scenario noisy')
    parser.add_argument('--ref', type=str, default='scenario/scenario_ref.txt',
                       help='Fichier scenario reference')
    parser.add_argument('--output', type=str, default='scenario_optimized.g2o',
                       help='Fichier g2o optimisé')
    parser.add_argument('--odom-info', type=float, default=500.0,
                       help='Information matrix pour odométrie')
    parser.add_argument('--loop-info', type=float, default=700.0,
                       help='Information matrix pour loop closures')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Nombre d\'itérations g2o')
    parser.add_argument('--loop-sampling', type=int, default=1,
                       help='Prendre 1 loop sur X (default: 1 = tous les loops). 0 = pas de loops')
    parser.add_argument('--no-viz', action='store_true',
                       help='Désactiver visualisation')
    args = parser.parse_args()

    print("="*70)
    print("OPTIMISATION DU SCENARIO AVEC G2O")
    print("="*70)
    print(f"  Input: {args.input}")
    print(f"  Reference: {args.ref}")
    print(f"  Odometry info: {args.odom_info}")
    print(f"  Loop closure info: {args.loop_info}")
    print(f"  Loop sampling: {args.loop_sampling} ({'tous les loops' if args.loop_sampling == 1 else 'aucun loop' if args.loop_sampling == 0 else f'1 loop sur {args.loop_sampling}'})")
    print(f"  G2O iterations: {args.iterations}")
    print("="*70)

    # Étape 1: Charger les données
    print(f"\n[1/6] Chargement des scénarios...")
    data_noisy = load_scenario(args.input)
    data_ref = load_scenario(args.ref)
    print(f"  Noisy: {len(data_noisy)} poses")
    print(f"  Reference: {len(data_ref)} poses")

    # Extraire les loop pairs avant sampling pour visualisation
    print(f"\n[2/6] Extraction des loop closures...")
    all_loop_pairs = []
    for i, d in enumerate(data_noisy):
        for loop_id in d['loop_ids']:
            if i < loop_id and loop_id < len(data_noisy):
                all_loop_pairs.append((i, loop_id))

    print(f"  {len(all_loop_pairs)} loop closures trouvées")

    # Appliquer le sampling
    if args.loop_sampling == 0:
        sampled_loop_pairs = []
        print(f"  Loop sampling: AUCUN loop closure utilisé (--loop-sampling 0)")
    elif args.loop_sampling > 1:
        sampled_loop_pairs = all_loop_pairs[::args.loop_sampling]
        print(f"  Loop sampling: {len(sampled_loop_pairs)} loops utilisés (1 sur {args.loop_sampling})")
    else:
        sampled_loop_pairs = all_loop_pairs
        print(f"  Loop sampling: TOUS les {len(sampled_loop_pairs)} loops utilisés")

    # Visualisation du scenario noisy avec loops AVANT optimisation
    if not args.no_viz:
        print(f"\n  Visualisation du scenario noisy avec {len(sampled_loop_pairs)} loop closures...")
        visualize_single(data_noisy,
                        f'Scenario NOISY (avant optimisation)\n{len(data_noisy)} poses, {len(sampled_loop_pairs)} loop closures',
                        color='red',
                        loop_pairs=sampled_loop_pairs)

    # Étape 3: Convertir noisy en g2o (loops depuis reference)
    noisy_g2o = 'scenario_noisy_input.g2o'
    print(f"\n[3/6] Conversion de {args.input} en g2o...")
    print(f"      Loop closures calculées depuis reference (identité)")

    write_g2o_from_scenario(args.input, noisy_g2o, args.odom_info, args.loop_info,
                            ref_file_for_loops=args.ref, loop_sampling=args.loop_sampling)

    # Étape 4: Convertir reference en g2o
    ref_g2o = 'scenario_ref_input.g2o'
    print(f"\n[4/7] Conversion de {args.ref} en g2o...")
    write_g2o_from_scenario(args.ref, ref_g2o, args.odom_info, args.loop_info,
                            loop_sampling=args.loop_sampling)

    # Étape 5: Optimiser
    print(f"\n[5/7] Optimisation avec g2o ({args.iterations} itérations)...")
    cmd = f"g2o -i {args.iterations} -o {args.output} {noisy_g2o}"
    print(f"  Commande: {cmd}")
    ret = os.system(cmd)

    if ret != 0:
        print(f"\n❌ ERREUR: g2o a échoué")
        return False

    print(f"✓ Optimisation terminée!")

    # Étape 6: Lire résultats
    print(f"\n[6/7] Lecture des résultats...")
    data_opt = read_g2o_optimized(args.output)
    print(f"  {len(data_opt)} poses optimisées")

    # Sauvegarder en txt
    output_txt = args.output.replace('.g2o', '.txt')
    with open(output_txt, 'w') as f:
        f.write("pose_id,tx,ty,tz,qx,qy,qz,qw\n")
        for d in data_opt:
            f.write(f"{d['pose_id']},{d['tx']:.9f},{d['ty']:.9f},{d['tz']:.9f},"
                   f"{d['qx']:.9f},{d['qy']:.9f},{d['qz']:.9f},{d['qw']:.9f}\n")

    print(f"✓ Sauvegardé: {output_txt}")

    # Étape 7: Métriques
    print(f"\n[7/7] Calcul des métriques...")

    # Debug: afficher quelques quaternions pour vérifier
    print("\n  [DEBUG] Vérification des quaternions:")
    print(f"    Noisy pose 0: quat = [{data_noisy[0]['qx']:.4f}, {data_noisy[0]['qy']:.4f}, {data_noisy[0]['qz']:.4f}, {data_noisy[0]['qw']:.4f}]")
    print(f"    Ref pose 0:   quat = [{data_ref[0]['qx']:.4f}, {data_ref[0]['qy']:.4f}, {data_ref[0]['qz']:.4f}, {data_ref[0]['qw']:.4f}]")
    print(f"    Noisy pose 100: quat = [{data_noisy[100]['qx']:.4f}, {data_noisy[100]['qy']:.4f}, {data_noisy[100]['qz']:.4f}, {data_noisy[100]['qw']:.4f}]")
    print(f"    Ref pose 100:   quat = [{data_ref[100]['qx']:.4f}, {data_ref[100]['qy']:.4f}, {data_ref[100]['qz']:.4f}, {data_ref[100]['qw']:.4f}]")

    pos_err_noisy, ori_err_noisy, stats_noisy = compute_pose_errors_3d(data_noisy, data_ref)
    pos_err_opt, ori_err_opt, stats_opt = compute_pose_errors_3d(data_opt, data_ref)

    print_metrics(stats_noisy, stats_opt)

    # Visualisations
    if not args.no_viz:
        print("\n" + "="*70)
        print("VISUALISATIONS")
        print("="*70)

        print("\n[1/4] Comparaison Reference vs Optimized...")
        visualize_two(data_ref, data_opt)

        print("\n[2/4] Comparaison des trois trajectoires (séparées)...")
        visualize_three(data_ref, data_opt, data_noisy)

        print("\n[3/4] Comparaison des trois trajectoires (SUPERPOSÉES)...")
        visualize_three_overlayed(data_ref, data_opt, data_noisy)

        print("\n[4/4] Graphiques d'erreurs...")
        plot_errors(pos_err_noisy, ori_err_noisy, pos_err_opt, ori_err_opt,
                   stats_noisy, stats_opt)

    print("\n" + "="*70)
    print("✓ TERMINÉ!")
    print("="*70)
    print(f"\nFichiers générés:")
    print(f"  - {noisy_g2o}")
    print(f"  - {ref_g2o}")
    print(f"  - {args.output}")
    print(f"  - {output_txt}")

    return True

if __name__ == '__main__':
    main()
