#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Script pour estimer les poses relatives entre des paires d'images query/loop
avec MapAnything et extraire les scores de confiance pour filtrer les mauvaises loops.

Usage:
    python scripts/estimate_loop_poses.py --image_folder /path/to/folder --output results.csv

Le dossier doit contenir:
    - query_1.jpg, query_2.jpg, ...
    - loop_1.jpg, loop_2.jpg, ...
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def find_image_pairs(image_folder: str) -> List[Tuple[str, str, int]]:
    """
    Trouve toutes les paires query/loop dans le dossier.

    Args:
        image_folder: Chemin vers le dossier contenant les images

    Returns:
        Liste de tuples (query_path, loop_path, pair_id)
    """
    folder = Path(image_folder)

    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Trouver toutes les images query et loop
    query_files = {}
    loop_files = {}

    for file_path in folder.iterdir():
        if file_path.suffix.lower() not in image_extensions:
            continue

        filename = file_path.stem

        # Parser query_N
        if filename.startswith('query_'):
            try:
                pair_id = int(filename.split('_')[1])
                query_files[pair_id] = str(file_path)
            except (IndexError, ValueError):
                print(f"Warning: Impossible de parser {filename}")

        # Parser loop_N
        elif filename.startswith('loop_'):
            try:
                pair_id = int(filename.split('_')[1])
                loop_files[pair_id] = str(file_path)
            except (IndexError, ValueError):
                print(f"Warning: Impossible de parser {filename}")

    # Créer les paires
    pairs = []
    common_ids = sorted(set(query_files.keys()) & set(loop_files.keys()))

    for pair_id in common_ids:
        pairs.append((query_files[pair_id], loop_files[pair_id], pair_id))

    return pairs


def compute_relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Calcule la pose relative de pose2 par rapport à pose1.

    Args:
        pose1: Matrice 4x4 de pose cam2world pour l'image 1
        pose2: Matrice 4x4 de pose cam2world pour l'image 2

    Returns:
        Matrice 4x4 de transformation relative (pose de 2 dans le référentiel de 1)
    """
    # Pose relative = pose1^-1 @ pose2
    # Cela donne la transformation de la caméra 2 dans le référentiel de la caméra 1
    pose1_inv = np.linalg.inv(pose1)
    relative_pose = pose1_inv @ pose2
    return relative_pose


def pose_to_dict(pose: np.ndarray) -> Dict:
    """
    Convertit une matrice de pose 4x4 en dictionnaire avec rotation (quaternion) et translation.

    Args:
        pose: Matrice 4x4 de transformation

    Returns:
        Dictionnaire avec 'translation' (tx, ty, tz) et 'rotation_matrix' (3x3)
    """
    translation = pose[:3, 3]
    rotation = pose[:3, :3]

    return {
        'tx': float(translation[0]),
        'ty': float(translation[1]),
        'tz': float(translation[2]),
        'r11': float(rotation[0, 0]),
        'r12': float(rotation[0, 1]),
        'r13': float(rotation[0, 2]),
        'r21': float(rotation[1, 0]),
        'r22': float(rotation[1, 1]),
        'r23': float(rotation[1, 2]),
        'r31': float(rotation[2, 0]),
        'r32': float(rotation[2, 1]),
        'r33': float(rotation[2, 2]),
    }


def compute_confidence_scores(predictions: List[Dict]) -> Tuple[float, float, float]:
    """
    Calcule des statistiques de confiance à partir des prédictions MapAnything.

    Args:
        predictions: Liste des prédictions pour chaque vue [view1, view2]

    Returns:
        Tuple (conf_mean_view1, conf_mean_view2, conf_mean_combined)
    """
    confidences = []

    for pred in predictions:
        if 'conf' in pred:
            conf = pred['conf'].cpu().numpy()  # Shape: (B, H, W) ou (B, H, W, 1)

            # S'assurer que c'est 3D (B, H, W)
            if conf.ndim == 4:
                conf = conf.squeeze(-1)

            # Prendre seulement les zones valides (masque si disponible)
            if 'mask' in pred:
                mask = pred['mask'].cpu().numpy().squeeze(-1).astype(bool)
                valid_conf = conf[mask]
            else:
                valid_conf = conf.flatten()

            if len(valid_conf) > 0:
                mean_conf = float(np.mean(valid_conf))
                confidences.append(mean_conf)
            else:
                confidences.append(0.0)
        else:
            confidences.append(None)

    # Retourner les scores
    conf_view1 = confidences[0] if len(confidences) > 0 else None
    conf_view2 = confidences[1] if len(confidences) > 1 else None

    # Score combiné (moyenne des deux vues)
    valid_confs = [c for c in confidences if c is not None]
    conf_combined = float(np.mean(valid_confs)) if valid_confs else None

    return conf_view1, conf_view2, conf_combined


def main():
    parser = argparse.ArgumentParser(
        description="Estime les poses relatives entre paires query/loop avec MapAnything"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Dossier contenant les images query_N.jpg et loop_N.jpg"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loop_poses.csv",
        help="Fichier CSV de sortie pour les résultats (default: loop_poses.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/map-anything",
        help="Modèle MapAnything à utiliser (default: facebook/map-anything)"
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Utiliser le modèle Apache 2.0 (facebook/map-anything-apache)"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        help="Utiliser l'inférence économe en mémoire"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Seuil de confiance pour filtrer les mauvaises loops (optionnel)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forcer l'utilisation du CPU au lieu du GPU (plus lent mais sans CUDA)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Nombre de paires à traiter avant de vider le cache GPU (default: 1). Augmenter peut accélérer mais consomme plus de mémoire"
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Nombre maximum de paires à traiter (utile pour tester sur un sous-ensemble)"
    )

    args = parser.parse_args()

    # Vérifier que le dossier existe
    if not os.path.exists(args.image_folder):
        print(f"Erreur: Le dossier {args.image_folder} n'existe pas")
        sys.exit(1)

    # Trouver les paires d'images
    print(f"Recherche des paires d'images dans: {args.image_folder}")
    pairs = find_image_pairs(args.image_folder)

    if not pairs:
        print("Erreur: Aucune paire query/loop trouvée dans le dossier")
        print("Format attendu: query_1.jpg, loop_1.jpg, query_2.jpg, loop_2.jpg, ...")
        sys.exit(1)

    print(f"Trouvé {len(pairs)} paires d'images")

    # Limiter le nombre de paires si demandé
    if args.max_pairs is not None and args.max_pairs < len(pairs):
        print(f"Limitation à {args.max_pairs} paires (option --max_pairs)")
        pairs = pairs[:args.max_pairs]

    # Configuration du device
    if args.cpu:
        device = "cpu"
        print(f"Mode CPU forcé (option --cpu)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")

    if device == "cpu":
        print("⚠️  ATTENTION: Le traitement sur CPU est beaucoup plus lent (10-50x)")
        print("   Attendez-vous à plusieurs minutes par paire d'images")

    # Afficher la configuration du batch
    if args.batch_size > 1:
        print(f"Traitement par batch de {args.batch_size} paires")
        print("⚠️  Un batch_size > 1 peut accélérer le traitement mais consomme plus de mémoire")

    # Charger le modèle
    if args.apache:
        model_name = "facebook/map-anything-apache"
    else:
        model_name = args.model

    print(f"Chargement du modèle {model_name}...")
    model = MapAnything.from_pretrained(model_name).to(device)
    model.eval()

    # Préparer le fichier CSV de sortie
    csv_file = open(args.output, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # En-têtes CSV
    headers = [
        'pair_id', 'query_image', 'loop_image',
        'tx', 'ty', 'tz',  # Translation de la pose relative
        'r11', 'r12', 'r13',  # Rotation (matrice 3x3)
        'r21', 'r22', 'r23',
        'r31', 'r32', 'r33',
        'conf_query', 'conf_loop', 'conf_mean',  # Scores de confiance
        'is_good_loop',  # Basé sur le seuil de confiance
        'processing_time_s'  # Temps de traitement en secondes
    ]
    csv_writer.writerow(headers)

    # Traiter chaque paire
    print("\nTraitement des paires:")
    print("-" * 80)

    results = []
    processing_times = []

    for pair_idx, (query_path, loop_path, pair_id) in enumerate(pairs):
        # Démarrer le chronomètre pour cette paire
        pair_start_time = time.time()

        print(f"\nPaire {pair_id} ({pair_idx + 1}/{len(pairs)}):")
        print(f"  Query: {Path(query_path).name}")
        print(f"  Loop:  {Path(loop_path).name}")

        # Charger les deux images
        views = load_images([query_path, loop_path])

        if len(views) != 2:
            print(f"  ⚠️  Erreur: Impossible de charger les images, skip")
            continue

        # Faire l'inférence
        inference_start = time.time()
        print(f"  Inférence en cours...")
        with torch.no_grad():
            predictions = model.infer(
                views,
                memory_efficient_inference=args.memory_efficient,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,  # On veut la confiance brute
            )
        inference_time = time.time() - inference_start

        # Extraire les poses
        pose_query = predictions[0]['camera_poses'][0].cpu().numpy()  # (4, 4)
        pose_loop = predictions[1]['camera_poses'][0].cpu().numpy()   # (4, 4)

        # Calculer la pose relative
        relative_pose = compute_relative_pose(pose_query, pose_loop)
        pose_dict = pose_to_dict(relative_pose)

        # Extraire les scores de confiance
        conf_query, conf_loop, conf_mean = compute_confidence_scores(predictions)

        # Déterminer si c'est une bonne loop
        is_good_loop = True
        if args.confidence_threshold is not None and conf_mean is not None:
            is_good_loop = conf_mean >= args.confidence_threshold

        # Calculer le temps total pour cette paire
        pair_processing_time = time.time() - pair_start_time
        processing_times.append(pair_processing_time)

        # Afficher les résultats
        translation_norm = np.linalg.norm(relative_pose[:3, 3])
        print(f"  ✓ Pose relative calculée:")
        print(f"    - Translation: [{pose_dict['tx']:.3f}, {pose_dict['ty']:.3f}, {pose_dict['tz']:.3f}] (norm: {translation_norm:.3f})")

        if conf_mean is not None:
            print(f"  ✓ Confiance:")
            print(f"    - Query: {conf_query:.4f}")
            print(f"    - Loop:  {conf_loop:.4f}")
            print(f"    - Moyenne: {conf_mean:.4f}")

            if args.confidence_threshold is not None:
                status = "✓ BONNE" if is_good_loop else "✗ MAUVAISE"
                print(f"  → {status} loop (seuil: {args.confidence_threshold:.4f})")
        else:
            print(f"  ⚠️  Confiance non disponible (modèle ne fournit pas 'conf')")

        # Afficher le temps de traitement
        print(f"  ⏱️  Temps: {pair_processing_time:.2f}s (inférence: {inference_time:.2f}s)")

        # Écrire dans le CSV
        row = [
            pair_id,
            Path(query_path).name,
            Path(loop_path).name,
            pose_dict['tx'], pose_dict['ty'], pose_dict['tz'],
            pose_dict['r11'], pose_dict['r12'], pose_dict['r13'],
            pose_dict['r21'], pose_dict['r22'], pose_dict['r23'],
            pose_dict['r31'], pose_dict['r32'], pose_dict['r33'],
            conf_query if conf_query is not None else '',
            conf_loop if conf_loop is not None else '',
            conf_mean if conf_mean is not None else '',
            is_good_loop,
            f"{pair_processing_time:.3f}"
        ]
        csv_writer.writerow(row)
        csv_file.flush()

        results.append({
            'pair_id': pair_id,
            'conf_mean': conf_mean,
            'is_good_loop': is_good_loop,
            'processing_time': pair_processing_time
        })

        # Vider le cache GPU après chaque batch
        if device == "cuda" and args.batch_size > 0 and (pair_idx + 1) % args.batch_size == 0:
            torch.cuda.empty_cache()
            print(f"  🗑️  Cache GPU vidé (batch de {args.batch_size} complété)")

    # Fermer le fichier CSV
    csv_file.close()

    # Statistiques finales
    print("\n" + "=" * 80)
    print("RÉSUMÉ:")
    print("=" * 80)
    print(f"Paires traitées: {len(results)}")
    print(f"Résultats sauvegardés dans: {args.output}")

    if args.confidence_threshold is not None:
        good_loops = sum(1 for r in results if r['is_good_loop'])
        bad_loops = len(results) - good_loops
        print(f"\nFiltre de confiance (seuil = {args.confidence_threshold}):")
        print(f"  - Bonnes loops: {good_loops}")
        print(f"  - Mauvaises loops: {bad_loops}")

    # Distribution des confiances
    confs = [r['conf_mean'] for r in results if r['conf_mean'] is not None]
    if confs:
        print(f"\nDistribution de confiance:")
        print(f"  - Min:     {min(confs):.4f}")
        print(f"  - Max:     {max(confs):.4f}")
        print(f"  - Moyenne: {np.mean(confs):.4f}")
        print(f"  - Médiane: {np.median(confs):.4f}")

    # Statistiques de temps
    if processing_times:
        total_time = sum(processing_times)
        avg_time = np.mean(processing_times)
        print(f"\nTemps de traitement:")
        print(f"  - Temps total:   {total_time:.2f}s")
        print(f"  - Temps moyen:   {avg_time:.2f}s/paire")
        print(f"  - Temps min:     {min(processing_times):.2f}s")
        print(f"  - Temps max:     {max(processing_times):.2f}s")

    print("\n✓ Terminé avec succès!")


if __name__ == "__main__":
    main()
