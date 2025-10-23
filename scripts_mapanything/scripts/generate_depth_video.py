#!/usr/bin/env python3
"""
Script pour faire l'inférence MapAnything sur un dossier d'images et générer une vidéo des depth maps.

Usage:
    python scripts/generate_depth_video.py --input_folder /path/to/images --output_video /path/to/output.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Configuration pour une meilleure efficacité mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def create_depth_colormap(depth: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
    """
    Convertit une depth map en image colorée pour la visualisation.
    
    Args:
        depth: Depth map (H, W)
        colormap: Colormap OpenCV à utiliser
        
    Returns:
        Image colorée (H, W, 3) en uint8
    """
    # Normaliser la depth map entre 0 et 1
    depth_normalized = depth.copy()
    
    # Ignorer les valeurs nulles/infinies
    valid_mask = (depth_normalized > 0) & np.isfinite(depth_normalized)
    
    if valid_mask.sum() > 0:
        min_val = depth_normalized[valid_mask].min()
        max_val = depth_normalized[valid_mask].max()
        
        if max_val > min_val:
            depth_normalized[valid_mask] = (depth_normalized[valid_mask] - min_val) / (max_val - min_val)
        else:
            depth_normalized[valid_mask] = 0.5
    
    # Convertir en uint8
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Appliquer la colormap
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    
    return depth_colored


def get_image_files(folder_path: str) -> List[str]:
    """
    Récupère tous les fichiers d'images dans un dossier.
    
    Args:
        folder_path: Chemin vers le dossier d'images
        
    Returns:
        Liste triée des chemins d'images
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(description="Générer une vidéo de depth maps avec MapAnything")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les images d'entrée")
    parser.add_argument("--output_video", required=True, help="Chemin de la vidéo de sortie (.mp4)")
    parser.add_argument("--model", default="facebook/map-anything", 
                       help="Modèle à utiliser (facebook/map-anything ou facebook/map-anything-apache)")
    parser.add_argument("--fps", type=int, default=10, help="FPS de la vidéo de sortie")
    parser.add_argument("--colormap", default="plasma", 
                       choices=['plasma', 'viridis', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'],
                       help="Colormap pour visualiser les depth maps")
    parser.add_argument("--memory_efficient", action="store_true", 
                       help="Utiliser l'inférence économe en mémoire (plus lent mais supporte plus d'images)")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Nombre maximum d'images à traiter")
    parser.add_argument("--save_individual", action="store_true",
                       help="Sauvegarder aussi les depth maps individuelles")
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Nombre d'images à traiter par batch (pour éviter les problèmes de mémoire)")
    
    args = parser.parse_args()
    
    # Vérifier que le dossier d'entrée existe
    if not os.path.exists(args.input_folder):
        print(f"Erreur: Le dossier {args.input_folder} n'existe pas")
        sys.exit(1)
    
    # Récupérer les fichiers d'images
    image_files = get_image_files(args.input_folder)
    if not image_files:
        print(f"Erreur: Aucun fichier d'image trouvé dans {args.input_folder}")
        sys.exit(1)
    
    # Limiter le nombre d'images si spécifié
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Traitement de {len(image_files)} images par batches de {args.batch_size}...")
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model}...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    
    # Traiter les images par batches
    all_predictions = []
    num_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
    
    print(f"Traitement en {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"Batch {batch_idx + 1}/{num_batches}: images {start_idx + 1}-{end_idx} ({len(batch_files)} images)")
        
        # Charger et préprocesser le batch d'images
        print(f"  Chargement des images du batch...")
        views = load_images(batch_files)
        
        # Faire l'inférence sur le batch
        print(f"  Inférence en cours...")
        with torch.no_grad():
            batch_predictions = model.infer(
                views,
                memory_efficient_inference=args.memory_efficient,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )
        
        # Ajouter les prédictions à la liste globale
        all_predictions.extend(batch_predictions)
        
        # Libérer la mémoire
        del views, batch_predictions
        torch.cuda.empty_cache()
        
        print(f"  Batch {batch_idx + 1} terminé")
    
    print(f"Inférence terminée pour {len(all_predictions)} images")
    predictions = all_predictions
    
    # Mapper les colormaps
    colormap_dict = {
        'plasma': cv2.COLORMAP_PLASMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL,
        'spring': cv2.COLORMAP_SPRING,
        'summer': cv2.COLORMAP_SUMMER,
        'autumn': cv2.COLORMAP_AUTUMN,
        'winter': cv2.COLORMAP_WINTER,
    }
    colormap = colormap_dict[args.colormap]
    
    # Préparer la sortie
    output_dir = Path(args.output_video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_individual:
        individual_dir = output_dir / "depth_maps"
        individual_dir.mkdir(exist_ok=True)
    
    # Traiter les depth maps et créer la vidéo
    print("Génération de la vidéo...")
    
    # Récupérer les dimensions de la première image pour la vidéo
    first_depth = predictions[0]["depth_z"].cpu().numpy().squeeze()
    depth_colored = create_depth_colormap(first_depth, colormap)
    height, width = depth_colored.shape[:2]
    
    # Créer le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))
    
    for i, pred in enumerate(tqdm(predictions, desc="Création vidéo")):
        # Extraire la depth map
        depth_z = pred["depth_z"].cpu().numpy().squeeze()  # (H, W)
        
        # Créer l'image colorée
        depth_colored = create_depth_colormap(depth_z, colormap)
        
        # Ajouter à la vidéo
        video_writer.write(depth_colored)
        
        # Sauvegarder individuellement si demandé
        if args.save_individual:
            filename = individual_dir / f"depth_{i:04d}.png"
            cv2.imwrite(str(filename), depth_colored)
    
    video_writer.release()
    
    print(f"Vidéo sauvegardée: {args.output_video}")
    if args.save_individual:
        print(f"Depth maps individuelles sauvegardées dans: {individual_dir}")
    
    # Afficher quelques statistiques
    print("\nStatistiques:")
    print(f"- Nombre d'images traitées: {len(predictions)}")
    print(f"- Résolution: {width}x{height}")
    print(f"- FPS: {args.fps}")
    print(f"- Colormap: {args.colormap}")
    
    # Calculer quelques statistiques sur les depth maps
    all_depths = []
    for pred in predictions:
        depth = pred["depth_z"].cpu().numpy().squeeze()
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.extend(valid_depth.tolist())
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur min: {all_depths.min():.2f}")
        print(f"- Profondeur max: {all_depths.max():.2f}")
        print(f"- Profondeur moyenne: {all_depths.mean():.2f}")


if __name__ == "__main__":
    main()