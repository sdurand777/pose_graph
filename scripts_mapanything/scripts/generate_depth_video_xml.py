#!/usr/bin/env python3
"""
Version du générateur de vidéo depth maps qui utilise la calibration XML avec redimensionnement automatique.

Usage:
    python scripts/generate_depth_video_xml.py --input_folder /path/to/images --xml_calib /path/to/calib.xml --camera_id lrl --output_video /path/to/output.mp4
"""

import argparse
import gc
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Configuration pour une meilleure efficacité mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def parse_xml_calibration(xml_path: str) -> Dict:
    """Parse un fichier de calibration XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    calibration = {}
    
    # Parser les caméras
    cameras = root.find('cameras')
    for camera in cameras.findall('camera'):
        camera_id = camera.get('id')
        
        # Taille d'image
        imagesize = camera.find('imagesize')
        width = int(imagesize.find('width').text)
        height = int(imagesize.find('height').text)
        
        # Intrinsèques
        intrinsics = camera.find('intrinsics')
        fx = float(intrinsics.find('fx').text)
        fy = float(intrinsics.find('fy').text)
        cx = float(intrinsics.find('cx').text)
        cy = float(intrinsics.find('cy').text)
        
        # Distortion
        distortion = intrinsics.find('distortion')
        k1 = float(distortion.find('k1').text)
        k2 = float(distortion.find('k2').text)
        k3 = float(distortion.find('k3').text)
        p1 = float(distortion.find('p1').text)
        p2 = float(distortion.find('p2').text)
        
        calibration[camera_id] = {
            'image_size': [width, height],
            'intrinsics': [fx, fy, cx, cy],
            'distortion': [k1, k2, p1, p2, k3]
        }
    
    return calibration


def scale_calibration_params(calib_params: Dict, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Dict:
    """Met à l'échelle les paramètres de calibration."""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    scaled_params = calib_params.copy()
    
    # Mettre à l'échelle les intrinsèques
    fx, fy, cx, cy = calib_params['intrinsics']
    scaled_params['intrinsics'] = [
        fx * scale_x,
        fy * scale_y,
        cx * scale_x,
        cy * scale_y
    ]
    
    scaled_params['distortion'] = calib_params['distortion'].copy()
    scaled_params['image_size'] = [target_w, target_h]
    
    return scaled_params


def create_intrinsics_matrix(intrinsics_list: List[float]) -> np.ndarray:
    """Crée une matrice d'intrinsèques 3x3."""
    fx, fy, cx, cy = intrinsics_list
    return np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=np.float32)


def undistort_image(image: np.ndarray, intrinsics: np.ndarray, distortion: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Corrige la distortion d'une image."""
    h, w = image.shape[:2]
    
    dist_coeffs = np.array(distortion, dtype=np.float32)
    
    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
        intrinsics, dist_coeffs, (w, h), alpha=1.0
    )
    
    image_undistorted = cv2.undistort(image, intrinsics, dist_coeffs, None, new_intrinsics)
    
    return image_undistorted, new_intrinsics


def create_depth_colormap(depth: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
    """Convertit une depth map en image colorée."""
    depth_normalized = depth.copy()
    
    valid_mask = (depth_normalized > 0) & np.isfinite(depth_normalized)
    
    if valid_mask.sum() > 0:
        min_val = depth_normalized[valid_mask].min()
        max_val = depth_normalized[valid_mask].max()
        
        if max_val > min_val:
            depth_normalized[valid_mask] = (depth_normalized[valid_mask] - min_val) / (max_val - min_val)
        else:
            depth_normalized[valid_mask] = 0.5
    
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    
    return depth_colored


def get_image_files(folder_path: str) -> List[str]:
    """Récupère tous les fichiers d'images dans un dossier."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def aggressive_cleanup():
    """Nettoyage agressif de la mémoire."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(0.1)


def process_single_image_with_calib(
    model, 
    image_path: str, 
    calib_params: Dict,
    memory_efficient: bool = True
) -> Optional[dict]:
    """Traite une seule image avec calibration XML."""
    try:
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convertir BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Adapter la calibration à la taille de l'image
        original_size = calib_params['image_size']
        current_size = [w, h]
        
        if current_size != original_size:
            scaled_params = scale_calibration_params(calib_params, original_size, current_size)
        else:
            scaled_params = calib_params
        
        # Créer la matrice d'intrinsèques
        intrinsics = create_intrinsics_matrix(scaled_params['intrinsics'])
        
        # Corriger la distortion
        img_undist, K_undist = undistort_image(img, intrinsics, scaled_params['distortion'])
        
        # Créer la vue pour MapAnything
        view = {
            "img": torch.from_numpy(img_undist).float(),
            "intrinsics": torch.from_numpy(K_undist).float(),
            "is_metric_scale": torch.tensor([False]),
        }
        
        views = [view]
        
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
        
        result = predictions[0] if predictions else None
        
        # Nettoyer immédiatement
        del views, predictions, img, img_undist
        aggressive_cleanup()
        
        return result
        
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {e}")
        aggressive_cleanup()
        return None


def save_progress(processed_indices: List[int], progress_file: str):
    """Sauvegarde le progrès."""
    with open(progress_file, 'w') as f:
        f.write('\n'.join(map(str, processed_indices)))


def load_progress(progress_file: str) -> List[int]:
    """Charge le progrès."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    return []


def main():
    parser = argparse.ArgumentParser(description="Générer une vidéo de depth maps avec calibration XML")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les images")
    parser.add_argument("--xml_calib", required=True, help="Fichier de calibration XML")
    parser.add_argument("--camera_id", required=True, choices=['lrl', 'lrr'], help="ID de la caméra (lrl ou lrr)")
    parser.add_argument("--output_video", required=True, help="Vidéo de sortie (.mp4)")
    parser.add_argument("--model", default="facebook/map-anything", help="Modèle MapAnything")
    parser.add_argument("--fps", type=int, default=10, help="FPS de la vidéo")
    parser.add_argument("--colormap", default="plasma", 
                       choices=['plasma', 'viridis', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'],
                       help="Colormap pour les depth maps")
    parser.add_argument("--memory_efficient", action="store_true", help="Mode économe en mémoire")
    parser.add_argument("--max_images", type=int, help="Nombre maximum d'images")
    parser.add_argument("--save_individual", action="store_true", help="Sauvegarder les depth maps individuelles")
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis le dernier arrêt")
    
    args = parser.parse_args()
    
    # Vérifications
    if not os.path.exists(args.input_folder):
        print(f"Erreur: Le dossier {args.input_folder} n'existe pas")
        sys.exit(1)
    
    if not os.path.exists(args.xml_calib):
        print(f"Erreur: Le fichier XML {args.xml_calib} n'existe pas")
        sys.exit(1)
    
    # Charger la calibration XML
    print(f"Chargement de la calibration XML depuis {args.xml_calib}")
    calib = parse_xml_calibration(args.xml_calib)
    
    if args.camera_id not in calib:
        print(f"Erreur: Caméra {args.camera_id} non trouvée dans la calibration")
        print(f"Caméras disponibles: {list(calib.keys())}")
        sys.exit(1)
    
    camera_params = calib[args.camera_id]
    print(f"Calibration {args.camera_id} chargée:")
    print(f"  - Taille originale: {camera_params['image_size'][0]}x{camera_params['image_size'][1]}")
    print(f"  - Intrinsèques: fx={camera_params['intrinsics'][0]:.1f}, fy={camera_params['intrinsics'][1]:.1f}")
    
    # Récupérer les images
    image_files = get_image_files(args.input_folder)
    if not image_files:
        print(f"Erreur: Aucune image trouvée dans {args.input_folder}")
        sys.exit(1)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Traitement de {len(image_files)} images avec calibration XML...")
    
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
    
    # Initialiser le writer vidéo
    video_writer = None
    all_depths = []
    
    # Traiter les images une par une
    for i, image_path in enumerate(tqdm(image_files, desc="Traitement des images")):
        # Vérifier si cette image a déjà été traitée
        if i in processed_indices:
            continue
        
        print(f"Traitement de l'image {i+1}/{len(image_files)}: {Path(image_path).name}")
        
        # Traiter l'image avec calibration
        pred = process_single_image_with_calib(model, image_path, camera_params, args.memory_efficient)
        
        if pred is None:
            print(f"Échec du traitement de l'image {i+1}, passage à la suivante")
            continue
        
        try:
            # Extraire la depth map
            depth_z = pred["depth_z"].cpu().numpy().squeeze()
            
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
    
    # Afficher les statistiques
    print("\n=== Statistiques ===")
    print(f"- Images traitées avec succès: {len(processed_indices)}/{len(image_files)}")
    print(f"- Calibration utilisée: {args.camera_id}")
    if video_writer:
        print(f"- Résolution: {width}x{height}")
        print(f"- FPS: {args.fps}")
    print(f"- Colormap: {args.colormap}")
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur globale: {all_depths.min():.2f}-{all_depths.max():.2f}m (moyenne: {all_depths.mean():.2f}m)")
    
    # Nettoyer le fichier de progrès si tout s'est bien passé
    if len(processed_indices) == len(image_files):
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print("\nTraitement terminé avec succès!")
    else:
        print(f"\nTraitement interrompu. Utilisez --resume pour reprendre.")


if __name__ == "__main__":
    main()