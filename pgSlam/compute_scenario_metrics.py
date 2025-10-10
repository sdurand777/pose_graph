#!/usr/bin/env python3
"""
Script pour calculer les métriques d'erreur entre les scénarios
Compare noisy et optimized par rapport à reference (ground truth)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from g2o_io_scenario import load_scenario

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

def compute_pose_errors_3d(data_test, data_ref):
    """
    Calcule les erreurs de position et d'orientation entre deux trajectoires 3D

    Returns:
        position_errors: Array des erreurs de position (distance euclidienne en 3D)
        orientation_errors: Array des erreurs d'orientation (angle géodésique sur SO(3))
        stats: Dictionnaire avec RMSE, mean, max
    """
    n_poses = min(len(data_test), len(data_ref))

    position_errors = np.zeros(n_poses)
    orientation_errors = np.zeros(n_poses)

    for i in range(n_poses):
        # Erreur de position (distance euclidienne 3D)
        pos_test = np.array([data_test[i]['tx'], data_test[i]['ty'], data_test[i]['tz']])
        pos_ref = np.array([data_ref[i]['tx'], data_ref[i]['ty'], data_ref[i]['tz']])
        position_errors[i] = np.linalg.norm(pos_test - pos_ref)

        # Erreur d'orientation (angle géodésique)
        r_test = Rotation.from_quat([data_test[i]['qx'], data_test[i]['qy'],
                                     data_test[i]['qz'], data_test[i]['qw']])
        r_ref = Rotation.from_quat([data_ref[i]['qx'], data_ref[i]['qy'],
                                    data_ref[i]['qz'], data_ref[i]['qw']])

        # Distance géodésique sur SO(3)
        r_diff = r_ref.inv() * r_test
        orientation_errors[i] = r_diff.magnitude()  # Retourne l'angle en radians

    # Statistiques
    stats = {
        'position': {
            'rmse': np.sqrt(np.mean(position_errors ** 2)),
            'mean': np.mean(position_errors),
            'max': np.max(position_errors),
            'median': np.median(position_errors)
        },
        'orientation': {
            'rmse': np.sqrt(np.mean(orientation_errors ** 2)),
            'mean': np.mean(orientation_errors),
            'max': np.max(orientation_errors),
            'median': np.median(orientation_errors)
        }
    }

    return position_errors, orientation_errors, stats

def print_metrics_comparison(stats_noisy, stats_opt, title="MÉTRIQUES D'ERREUR"):
    """Affiche une comparaison formatée des métriques"""
    print("\n" + "="*70)
    print(title)
    print("="*70)

    print("\nERREUR DE POSITION (mètres):")
    print(f"{'Métrique':<15} {'Noisy':<15} {'Optimized':<15} {'Amélioration':<15}")
    print("-"*70)

    for metric in ['rmse', 'mean', 'max', 'median']:
        noisy_val = stats_noisy['position'][metric]
        opt_val = stats_opt['position'][metric]
        improvement = ((noisy_val - opt_val) / noisy_val * 100) if noisy_val > 0 else 0

        metric_name = metric.upper()
        print(f"{metric_name:<15} {noisy_val:<15.4f} {opt_val:<15.4f} {improvement:<15.2f}%")

    print("\nERREUR D'ORIENTATION (degrés):")
    print(f"{'Métrique':<15} {'Noisy':<15} {'Optimized':<15} {'Amélioration':<15}")
    print("-"*70)

    for metric in ['rmse', 'mean', 'max', 'median']:
        noisy_val = np.rad2deg(stats_noisy['orientation'][metric])
        opt_val = np.rad2deg(stats_opt['orientation'][metric])
        improvement = ((noisy_val - opt_val) / noisy_val * 100) if noisy_val > 0 else 0

        metric_name = metric.upper()
        print(f"{metric_name:<15} {noisy_val:<15.4f} {opt_val:<15.4f} {improvement:<15.2f}%")

    print("="*70)

def plot_errors(pos_errors_noisy, ori_errors_noisy,
                pos_errors_opt, ori_errors_opt,
                stats_noisy, stats_opt):
    """Affiche les graphiques d'erreurs par pose"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    n_poses = len(pos_errors_noisy)
    pose_indices = np.arange(n_poses)

    # Plot 1: Erreurs de position
    ax1.plot(pose_indices, pos_errors_noisy, 'r-', linewidth=1.5, alpha=0.7,
            label=f'Noisy (RMSE: {stats_noisy["position"]["rmse"]:.3f}m)')
    ax1.plot(pose_indices, pos_errors_opt, 'b-', linewidth=1.5, alpha=0.7,
            label=f'Optimized (RMSE: {stats_opt["position"]["rmse"]:.3f}m)')

    ax1.axhline(y=stats_noisy['position']['rmse'], color='red', linestyle='--',
               linewidth=1, alpha=0.5, label='RMSE Noisy')
    ax1.axhline(y=stats_opt['position']['rmse'], color='blue', linestyle='--',
               linewidth=1, alpha=0.5, label='RMSE Optimized')

    ax1.set_xlabel('Pose Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Erreur de Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Erreur de Position par Pose (par rapport à Reference)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Erreurs d'orientation
    ori_errors_noisy_deg = np.rad2deg(ori_errors_noisy)
    ori_errors_opt_deg = np.rad2deg(ori_errors_opt)
    rmse_noisy_deg = np.rad2deg(stats_noisy['orientation']['rmse'])
    rmse_opt_deg = np.rad2deg(stats_opt['orientation']['rmse'])

    ax2.plot(pose_indices, ori_errors_noisy_deg, 'r-', linewidth=1.5, alpha=0.7,
            label=f'Noisy (RMSE: {rmse_noisy_deg:.3f}°)')
    ax2.plot(pose_indices, ori_errors_opt_deg, 'b-', linewidth=1.5, alpha=0.7,
            label=f'Optimized (RMSE: {rmse_opt_deg:.3f}°)')

    ax2.axhline(y=rmse_noisy_deg, color='red', linestyle='--',
               linewidth=1, alpha=0.5, label='RMSE Noisy')
    ax2.axhline(y=rmse_opt_deg, color='blue', linestyle='--',
               linewidth=1, alpha=0.5, label='RMSE Optimized')

    ax2.set_xlabel('Pose Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Erreur d\'Orientation (degrés)', fontsize=12, fontweight='bold')
    ax2.set_title('Erreur d\'Orientation par Pose (par rapport à Reference)',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compute_metrics(noisy_file='scenario/scenario_noisy.txt',
                   optimized_file='scenario_optimized.txt',
                   ref_file='scenario/scenario_ref.txt',
                   plot=True):
    """
    Calcule et affiche les métriques d'erreur
    """
    print("Chargement des scénarios...")

    # Charger les données
    data_noisy = load_scenario(noisy_file)
    data_opt = load_optimized_txt(optimized_file)
    data_ref = load_scenario(ref_file)

    print(f"  Noisy: {len(data_noisy)} poses")
    print(f"  Optimized: {len(data_opt)} poses")
    print(f"  Reference: {len(data_ref)} poses")

    # Calculer les erreurs pour noisy
    print("\nCalcul des erreurs pour scenario noisy...")
    pos_errors_noisy, ori_errors_noisy, stats_noisy = compute_pose_errors_3d(data_noisy, data_ref)

    # Calculer les erreurs pour optimized
    print("Calcul des erreurs pour scenario optimized...")
    pos_errors_opt, ori_errors_opt, stats_opt = compute_pose_errors_3d(data_opt, data_ref)

    # Afficher les métriques
    print_metrics_comparison(stats_noisy, stats_opt)

    # Afficher les graphiques
    if plot:
        print("\nAffichage des graphiques d'erreurs...")
        plot_errors(pos_errors_noisy, ori_errors_noisy,
                   pos_errors_opt, ori_errors_opt,
                   stats_noisy, stats_opt)

    return stats_noisy, stats_opt

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculer les métriques d\'erreur')
    parser.add_argument('--noisy', type=str, default='scenario/scenario_noisy.txt',
                       help='Fichier scenario noisy')
    parser.add_argument('--optimized', type=str, default='scenario_optimized.txt',
                       help='Fichier scenario optimized')
    parser.add_argument('--ref', type=str, default='scenario/scenario_ref.txt',
                       help='Fichier scenario reference')
    parser.add_argument('--no-plot', action='store_true',
                       help='Désactiver l\'affichage des graphiques')
    args = parser.parse_args()

    compute_metrics(args.noisy, args.optimized, args.ref, plot=not args.no_plot)
