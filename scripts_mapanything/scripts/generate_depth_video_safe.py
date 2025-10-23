#!/usr/bin/env python3
"""
Version sécurisée du générateur de vidéo depth maps qui traite les images une par une
et permet la reprise en cas de crash.

Usage:
    python scripts/generate_depth_video_safe.py --input_folder /path/to/images --output_video /path/to/output.mp4
"""

import argparse
import gc
import os
import sys
import time
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


def aggressive_cleanup():
    """Nettoyage agressif de la mémoire."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(0.1)  # Petit délai pour laisser le temps au cleanup


def process_single_image(model, image_path: str, memory_efficient: bool = True) -> Optional[dict]:
    """
    Traite une seule image et retourne le résultat.
    
    Args:
        model: Modèle MapAnything
        image_path: Chemin vers l'image
        memory_efficient: Utiliser le mode économe en mémoire
        
    Returns:
        Dictionnaire avec les prédictions ou None en cas d'erreur
    """
    try:
        # Charger l'image
        views = load_images([image_path])
        
        # Faire l'inférence
        with torch.no_grad():
            predictions = model.infer(
                views,
                memory_efficient_inference=memory_efficient,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )
        
        # Récupérer le résultat (une seule image)
        result = predictions[0] if predictions else None
        
        # Nettoyer immédiatement
        del views, predictions
        aggressive_cleanup()
        
        return result
        
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {e}")
        aggressive_cleanup()
        return None


def save_progress(processed_indices: List[int], progress_file: str):
    """Sauvegarde le progrès dans un fichier."""
    with open(progress_file, 'w') as f:
        f.write('\n'.join(map(str, processed_indices)))


def load_progress(progress_file: str) -> List[int]:
    """Charge le progrès depuis un fichier."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    return []


def main():
    parser = argparse.ArgumentParser(description="Générer une vidéo de depth maps avec MapAnything (version sécurisée)")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les images d'entrée")
    parser.add_argument("--output_video", required=True, help="Chemin de la vidéo de sortie (.mp4)")
    parser.add_argument("--model", default="facebook/map-anything", 
                       help="Modèle à utiliser (facebook/map-anything ou facebook/map-anything-apache)")
    parser.add_argument("--fps", type=int, default=10, help="FPS de la vidéo de sortie")
    parser.add_argument("--colormap", default="plasma", 
                       choices=['plasma', 'viridis', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'],
                       help="Colormap pour visualiser les depth maps")
    parser.add_argument("--memory_efficient", action="store_true", 
                       help="Utiliser l'inférence économe en mémoire")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Nombre maximum d'images à traiter")
    parser.add_argument("--save_individual", action="store_true",
                       help="Sauvegarder aussi les depth maps individuelles")
    parser.add_argument("--resume", action="store_true",
                       help="Reprendre depuis le dernier arrêt")
    
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
    
    print(f"Traitement de {len(image_files)} images une par une...")
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model}...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    
    # Préparer la sortie
    output_dir = Path(args.output_video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_individual:
        individual_dir = output_dir / "depth_maps"
        individual_dir.mkdir(exist_ok=True)
    
    # Fichier de progrès
    progress_file = output_dir / "progress.txt"
    
    # Charger le progrès si demandé
    processed_indices = []
    if args.resume:
        processed_indices = load_progress(str(progress_file))
        print(f"Reprise depuis l'image {len(processed_indices) + 1}/{len(image_files)}")
    
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
    
    # Initialiser le writer vidéo (sera créé après la première image)
    video_writer = None
    
    # Traiter les images une par une
    all_depths = []
    
    for i, image_path in enumerate(tqdm(image_files, desc="Traitement des images")):
        # Vérifier si cette image a déjà été traitée
        if i in processed_indices:
            continue
        
        print(f"Traitement de l'image {i+1}/{len(image_files)}: {Path(image_path).name}")
        
        # Traiter l'image
        pred = process_single_image(model, image_path, args.memory_efficient)
        
        if pred is None:
            print(f"Échec du traitement de l'image {i+1}, passage à la suivante")
            continue
        
        try:
            # Extraire la depth map
            depth_z = pred["depth_z"].cpu().numpy().squeeze()  # (H, W)
            
            # Créer l'image colorée
            depth_colored = create_depth_colormap(depth_z, colormap)
            
            # Initialiser le writer vidéo avec la première image
            if video_writer is None:
                height, width = depth_colored.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))
                print(f"Vidéo initialisée: {width}x{height} à {args.fps} FPS")
            
            # Ajouter à la vidéo
            video_writer.write(depth_colored)
            
            # Sauvegarder individuellement si demandé
            if args.save_individual:
                filename = individual_dir / f"depth_{i:04d}.png"
                cv2.imwrite(str(filename), depth_colored)
            
            # Collecter les statistiques
            valid_depth = depth_z[depth_z > 0]
            if len(valid_depth) > 0:
                all_depths.extend(valid_depth.tolist())
            
            # Marquer comme traité
            processed_indices.append(i)
            
            # Nettoyer
            del pred, depth_z, depth_colored, valid_depth
            aggressive_cleanup()
            
            # Sauvegarder le progrès tous les 10 images
            if (i + 1) % 10 == 0:
                save_progress(processed_indices, str(progress_file))
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image {i+1}: {e}")
            continue
    
    # Finaliser la vidéo
    if video_writer:
        video_writer.release()
    
    # Sauvegarder le progrès final
    save_progress(processed_indices, str(progress_file))
    
    print(f"\nVidéo sauvegardée: {args.output_video}")
    if args.save_individual:
        print(f"Depth maps individuelles sauvegardées dans: {individual_dir}")
    
    # Afficher quelques statistiques
    print("\nStatistiques:")
    print(f"- Images traitées avec succès: {len(processed_indices)}/{len(image_files)}")
    if video_writer:
        print(f"- Résolution: {width}x{height}")
        print(f"- FPS: {args.fps}")
    print(f"- Colormap: {args.colormap}")
    
    # Calculer quelques statistiques sur les depth maps
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur min: {all_depths.min():.2f}")
        print(f"- Profondeur max: {all_depths.max():.2f}")
        print(f"- Profondeur moyenne: {all_depths.mean():.2f}")
    
    # Nettoyer le fichier de progrès si tout s'est bien passé
    if len(processed_indices) == len(image_files):
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print("\nTraitement terminé avec succès!")
    else:
        print(f"\nTraitement interrompu. Utilisez --resume pour reprendre.")


if __name__ == "__main__":
    main()