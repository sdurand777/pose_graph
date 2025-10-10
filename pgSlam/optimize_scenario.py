#!/usr/bin/env python3
"""
Script principal pour optimiser le scenario_noisy avec g2o
"""
import os
import argparse
from g2o_io_scenario import write_g2o_from_scenario, read_g2o_optimized, load_scenario
from visualize_scenario_optimization import visualize_optimization_results

def optimize_scenario(input_scenario='scenario/scenario_noisy.txt',
                     ref_scenario='scenario/scenario_ref.txt',
                     output_g2o='scenario_optimized.g2o',
                     odom_info=500.0,
                     loop_info=700.0,
                     g2o_iterations=50,
                     visualize=True):
    """
    Pipeline complet d'optimisation:
    1. Convertir scenario_noisy.txt et scenario_ref.txt en g2o
    2. Exécuter g2o pour optimiser
    3. Lire les résultats optimisés
    4. Visualiser les résultats
    """

    print("="*70)
    print("OPTIMISATION DU SCENARIO AVEC G2O")
    print("="*70)

    # Étape 1a: Convertir scenario noisy en g2o
    # IMPORTANT: Les loop closures sont calculées depuis les poses de référence (identité attendue)
    noisy_input_g2o = 'scenario_noisy_input.g2o'
    print(f"\n[1/4] Conversion de {input_scenario} en format g2o...")
    print(f"      Les loop closures seront calculées depuis {ref_scenario} (poses parfaites)")
    write_g2o_from_scenario(input_scenario, noisy_input_g2o, odom_info, loop_info, ref_file_for_loops=ref_scenario)

    # Étape 1b: Convertir scenario reference en g2o (pour vérification)
    ref_input_g2o = 'scenario_ref_input.g2o'
    print(f"\n[2/4] Conversion de {ref_scenario} en format g2o...")
    write_g2o_from_scenario(ref_scenario, ref_input_g2o, odom_info, loop_info)

    # Étape 2: Optimisation avec g2o
    print(f"\n[3/4] Optimisation avec g2o ({g2o_iterations} itérations)...")
    cmd = f"g2o -i {g2o_iterations} -o {output_g2o} {noisy_input_g2o}"
    print(f"  Commande: {cmd}")
    ret = os.system(cmd)

    if ret != 0:
        print(f"\n❌ ERREUR: g2o a échoué avec le code de retour {ret}")
        print("Vérifiez que g2o est installé: which g2o")
        return False

    print(f"✓ Optimisation terminée!")

    # Étape 3: Lire les résultats
    print(f"\n[4/4] Lecture des poses optimisées depuis {output_g2o}...")
    optimized_data = read_g2o_optimized(output_g2o)
    print(f"  {len(optimized_data)} poses optimisées chargées")

    # Sauvegarder au format texte pour faciliter la visualisation
    output_txt = output_g2o.replace('.g2o', '.txt')
    print(f"\nSauvegarde des poses optimisées dans {output_txt}...")
    with open(output_txt, 'w') as f:
        f.write("pose_id,tx,ty,tz,qx,qy,qz,qw\n")
        for d in optimized_data:
            f.write(f"{d['pose_id']},{d['tx']:.9f},{d['ty']:.9f},{d['tz']:.9f},"
                   f"{d['qx']:.9f},{d['qy']:.9f},{d['qz']:.9f},{d['qw']:.9f}\n")

    print(f"✓ Poses optimisées sauvegardées dans {output_txt}")

    print("\n" + "="*70)
    print("OPTIMISATION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print(f"\nFichiers générés:")
    print(f"  - {noisy_input_g2o} (scenario noisy en format g2o)")
    print(f"  - {ref_input_g2o} (scenario reference en format g2o)")
    print(f"  - {output_g2o} (résultat optimisé en format g2o)")
    print(f"  - {output_txt} (poses optimisées en format texte)")

    # Étape 4: Visualisation
    if visualize:
        print("\n" + "="*70)
        print("VISUALISATION DES RÉSULTATS")
        print("="*70)
        visualize_optimization_results(input_scenario, output_txt, ref_scenario)

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimiser le scenario noisy avec g2o')
    parser.add_argument('--input', type=str, default='scenario/scenario_noisy.txt',
                       help='Fichier de scénario bruité en entrée')
    parser.add_argument('--ref', type=str, default='scenario/scenario_ref.txt',
                       help='Fichier de scénario de référence')
    parser.add_argument('--output', type=str, default='scenario_optimized.g2o',
                       help='Fichier g2o optimisé en sortie')
    parser.add_argument('--odom-info', type=float, default=500.0,
                       help='Information matrix value pour odométrie')
    parser.add_argument('--loop-info', type=float, default=700.0,
                       help='Information matrix value pour loop closures')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Nombre d\'itérations pour g2o')
    parser.add_argument('--no-viz', action='store_true',
                       help='Désactiver la visualisation automatique')
    args = parser.parse_args()

    success = optimize_scenario(
        input_scenario=args.input,
        ref_scenario=args.ref,
        output_g2o=args.output,
        odom_info=args.odom_info,
        loop_info=args.loop_info,
        g2o_iterations=args.iterations,
        visualize=not args.no_viz
    )

    if not success:
        exit(1)
