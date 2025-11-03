#!/usr/bin/env python3
"""
Script pour filtrer les loops validées et copier les images correspondantes.

Ce script lit le fichier CSV de résultats d'estimation de poses,
filtre les loops qui ont passé le test de confiance,
et crée un nouveau dossier avec uniquement les images validées.

Usage:
    python scripts/filter_validated_loops.py --csv loop_poses.csv --image_folder /path/to/images --output_folder validated_loops --min_confidence 0.6
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtrer et copier les loops validées selon la confiance"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Fichier CSV contenant les résultats d'estimation (loop_poses.csv)"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Dossier source contenant toutes les images query/loop"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="validated_loops",
        help="Dossier de sortie pour les images validées (défaut: validated_loops)"
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Seuil minimum de confiance pour garder une loop (défaut: 0.0 = tout garder)"
    )
    parser.add_argument(
        "--list_file",
        type=str,
        default="validated_loops.txt",
        help="Fichier texte de sortie listant les loops validées (défaut: validated_loops.txt)"
    )

    return parser.parse_args()


def read_validated_loops(csv_file: str, min_confidence: float) -> List[Tuple[str, str, str, float]]:
    """
    Lit le CSV et retourne les loops qui passent le seuil de confiance.

    Args:
        csv_file: Chemin vers le fichier CSV
        min_confidence: Seuil minimum de confiance

    Returns:
        Liste de tuples (pair_id, query_image, loop_image, confidence)
    """
    validated = []

    print(f"📖 Lecture du fichier CSV: {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            pair_id = row['pair_id']
            query_image = row['query_image']
            loop_image = row['loop_image']
            confidence = float(row['conf_mean'])  # Utiliser conf_mean comme confiance

            if confidence >= min_confidence:
                validated.append((pair_id, query_image, loop_image, confidence))

    print(f"✅ {len(validated)} loops trouvées avec confiance >= {min_confidence}")

    return validated




def copy_validated_images(
    validated_loops: List[Tuple[str, str, str, float]],
    image_folder: str,
    output_folder: str
) -> int:
    """
    Copie les images des loops validées vers le dossier de sortie.

    Args:
        validated_loops: Liste des loops validées (pair_id, query_image, loop_image, confidence)
        image_folder: Dossier source
        output_folder: Dossier de destination

    Returns:
        Nombre d'images copiées
    """
    image_folder_path = Path(image_folder)
    output_folder_path = Path(output_folder)

    # Créer le dossier de sortie s'il n'existe pas
    output_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Copie des images vers: {output_folder}")

    copied_count = 0
    missing_files = []

    for pair_id, query_image, loop_image, confidence in validated_loops:
        try:
            # Chemins source
            query_src = image_folder_path / query_image
            loop_src = image_folder_path / loop_image

            # Chemins destination
            query_dst = output_folder_path / query_image
            loop_dst = output_folder_path / loop_image

            # Copier les fichiers
            files_to_copy = [
                (query_src, query_dst, query_image),
                (loop_src, loop_dst, loop_image)
            ]

            for src, dst, name in files_to_copy:
                if src.exists():
                    if not dst.exists():  # Ne pas copier si déjà présent
                        shutil.copy2(src, dst)
                        copied_count += 1
                else:
                    missing_files.append(str(src))

        except Exception as e:
            print(f"⚠️  Erreur pour pair {pair_id}: {e}")

    if missing_files:
        print(f"\n⚠️  {len(missing_files)} fichiers manquants:")
        for f in missing_files[:10]:  # Afficher les 10 premiers
            print(f"   - {f}")
        if len(missing_files) > 10:
            print(f"   ... et {len(missing_files) - 10} autres")

    return copied_count


def write_validated_list(
    validated_loops: List[Tuple[str, str, str, float]],
    output_file: str
):
    """
    Écrit la liste des loops validées dans un fichier texte avec la confiance.

    Args:
        validated_loops: Liste des loops validées (pair_id, query_image, loop_image, confidence)
        output_file: Chemin du fichier de sortie
    """
    print(f"\n📝 Écriture de la liste dans: {output_file}")

    with open(output_file, 'w') as f:
        # En-tête
        f.write("# ======================================================================\n")
        f.write("# Loops Validées - Liste avec Confiance Moyenne\n")
        f.write("# ======================================================================\n")
        f.write(f"# Total loops validées: {len(validated_loops)}\n")
        f.write("#\n")
        f.write("# Format: pair_id | query_image -> loop_image | confiance_moyenne\n")
        f.write("# ======================================================================\n\n")

        # Trier par confiance décroissante (les meilleures en premier)
        sorted_loops = sorted(validated_loops, key=lambda x: x[3], reverse=True)

        # Écrire chaque loop avec un format lisible
        for pair_id, query_image, loop_image, confidence in sorted_loops:
            # Format aligné pour meilleure lisibilité
            f.write(f"Loop {pair_id:>3} | {query_image:25} -> {loop_image:25} | Conf: {confidence:.4f}\n")

        # Pied de page avec statistiques
        if validated_loops:
            confidences = [conf for _, _, _, conf in validated_loops]
            f.write("\n# ======================================================================\n")
            f.write("# Statistiques\n")
            f.write("# ======================================================================\n")
            f.write(f"# Confiance minimum:  {min(confidences):.4f}\n")
            f.write(f"# Confiance maximum:  {max(confidences):.4f}\n")
            f.write(f"# Confiance moyenne:  {sum(confidences)/len(confidences):.4f}\n")
            f.write("# ======================================================================\n")

    print(f"✅ Liste écrite: {len(validated_loops)} entrées (triées par confiance décroissante)")


def print_statistics(validated_loops: List[Tuple[str, str, str, float]]):
    """
    Affiche des statistiques sur les loops validées.

    Args:
        validated_loops: Liste des loops validées (pair_id, query_image, loop_image, confidence)
    """
    if not validated_loops:
        print("\n⚠️  Aucune loop validée")
        return

    confidences = [conf for _, _, _, conf in validated_loops]

    print("\n" + "="*70)
    print("📊 STATISTIQUES DES LOOPS VALIDÉES")
    print("="*70)
    print(f"Nombre total:        {len(validated_loops)}")
    print(f"Confiance min:       {min(confidences):.4f}")
    print(f"Confiance max:       {max(confidences):.4f}")
    print(f"Confiance moyenne:   {sum(confidences)/len(confidences):.4f}")
    print("="*70)


def main():
    args = parse_args()

    print("="*70)
    print("🔍 FILTRAGE DES LOOPS VALIDÉES")
    print("="*70)
    print(f"CSV source:          {args.csv}")
    print(f"Images source:       {args.image_folder}")
    print(f"Dossier sortie:      {args.output_folder}")
    print(f"Seuil confiance:     {args.min_confidence}")
    print(f"Fichier liste:       {args.list_file}")
    print("="*70)

    # Vérifier que les fichiers/dossiers existent
    if not os.path.exists(args.csv):
        print(f"❌ Erreur: Le fichier CSV {args.csv} n'existe pas")
        return 1

    if not os.path.exists(args.image_folder):
        print(f"❌ Erreur: Le dossier {args.image_folder} n'existe pas")
        return 1

    # Lire les loops validées
    validated_loops = read_validated_loops(args.csv, args.min_confidence)

    if not validated_loops:
        print("\n⚠️  Aucune loop ne passe le seuil de confiance")
        print(f"   Essayez de réduire --min_confidence (actuellement: {args.min_confidence})")
        return 0

    # Afficher les statistiques
    print_statistics(validated_loops)

    # Écrire la liste dans un fichier texte
    write_validated_list(validated_loops, args.list_file)

    # Copier les images
    copied_count = copy_validated_images(
        validated_loops,
        args.image_folder,
        args.output_folder
    )

    print(f"\n✅ {copied_count} images copiées dans {args.output_folder}")

    print("\n" + "="*70)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*70)
    print(f"📁 Images validées: {args.output_folder}/")
    print(f"📝 Liste: {args.list_file}")
    print("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
