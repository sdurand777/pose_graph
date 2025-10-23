#!/usr/bin/env python3
"""
Demo MapAnything avec calibration mono-caméra (intrinsèques + distortion)

Usage:
    python scripts/demo_mono_calibration.py --input_folder /path/to/images --calib_file /path/to/calib.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# Configuration pour une meilleure efficacité mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images, rgb
from mapanything.utils.viz import predictions_to_glb


def load_mono_calibration_file(calib_path: str) -> Dict:
    """
    Charge un fichier de calibration mono-caméra YAML.
    
    Format attendu:
    camera:
      intrinsics: [fx, fy, cx, cy]
      distortion: [k1, k2, p1, p2, k3]  # Optionnel
      image_size: [width, height]
    """
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
    return calib


def create_intrinsics_matrix(intrinsics_list: List[float]) -> np.ndarray:
    """Crée une matrice d'intrinsèques 3x3 depuis une liste [fx, fy, cx, cy]."""
    fx, fy, cx, cy = intrinsics_list
    return np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=np.float32)


def undistort_image(image: np.ndarray, intrinsics: np.ndarray, distortion: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Corrige la distortion d'une image et retourne l'image corrigée + nouvelles intrinsèques.
    
    Args:
        image: Image à corriger (H, W, 3)
        intrinsics: Matrice intrinsèques (3, 3)
        distortion: Coefficients de distortion [k1, k2, p1, p2, k3] ou None
        
    Returns:
        image_undistorted: Image corrigée
        new_intrinsics: Nouvelles intrinsèques après correction
    """
    if distortion is None or all(abs(d) < 1e-8 for d in distortion):
        # Pas de distortion
        return image, intrinsics
    
    h, w = image.shape[:2]
    
    # Convertir en coefficients OpenCV
    dist_coeffs = np.array(distortion, dtype=np.float32)
    
    # Calculer les nouvelles intrinsèques optimales
    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
        intrinsics, dist_coeffs, (w, h), alpha=1.0
    )
    
    # Corriger la distortion
    image_undistorted = cv2.undistort(image, intrinsics, dist_coeffs, None, new_intrinsics)
    
    return image_undistorted, new_intrinsics


def get_image_files(folder_path: str) -> List[str]:
    """Récupère tous les fichiers d'images dans un dossier."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def prepare_views_with_mono_calibration(
    image_paths: List[str], 
    calib: Dict,
    max_images: Optional[int] = None
) -> List[Dict]:
    """
    Prépare les vues avec calibration mono-caméra pour MapAnything.
    
    Args:
        image_paths: Liste des chemins des images
        calib: Dictionnaire de calibration
        max_images: Nombre maximum d'images à traiter
        
    Returns:
        Liste des vues préparées pour MapAnything
    """
    # Limiter le nombre d'images si spécifié
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Traitement de {len(image_paths)} images avec calibration")
    
    # Extraire les paramètres de calibration
    intrinsics = create_intrinsics_matrix(calib['camera']['intrinsics'])
    distortion = calib['camera'].get('distortion', None)
    
    print(f"Intrinsèques: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}, cx={intrinsics[0,2]:.1f}, cy={intrinsics[1,2]:.1f}")
    if distortion:
        print(f"Distortion: k1={distortion[0]:.4f}, k2={distortion[1]:.4f}, p1={distortion[2]:.4f}, p2={distortion[3]:.4f}")
    
    views = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Préparation de l'image {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        # Charger l'image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Erreur lors du chargement de l'image {i+1}, passage à la suivante")
            continue
        
        # Convertir BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Corriger la distortion si nécessaire
        img_undist, K_undist = undistort_image(img, intrinsics, distortion)
        
        # Créer la vue pour MapAnything
        view = {
            "img": torch.from_numpy(img_undist).float(),
            "intrinsics": torch.from_numpy(K_undist).float(),
            "is_metric_scale": torch.tensor([False]),  # MapAnything va estimer l'échelle
            "idx": torch.tensor([i]),
        }
        
        views.append(view)
    
    return views


def main():
    parser = argparse.ArgumentParser(description="Demo MapAnything avec calibration mono-caméra")
    parser.add_argument("--input_folder", required=True, help="Dossier des images")
    parser.add_argument("--calib_file", required=True, help="Fichier de calibration YAML")
    parser.add_argument("--output_glb", default="mono_reconstruction.glb", help="Fichier GLB de sortie")
    parser.add_argument("--model", default="facebook/map-anything", 
                       help="Modèle à utiliser (facebook/map-anything ou facebook/map-anything-apache)")
    parser.add_argument("--max_images", type=int, default=None, help="Nombre maximum d'images à traiter")
    parser.add_argument("--memory_efficient", action="store_true", help="Mode économe en mémoire")
    
    args = parser.parse_args()
    
    # Vérifications
    if not os.path.exists(args.input_folder):
        print(f"Erreur: Le dossier {args.input_folder} n'existe pas")
        sys.exit(1)
    
    if not os.path.exists(args.calib_file):
        print(f"Erreur: Le fichier de calibration {args.calib_file} n'existe pas")
        sys.exit(1)
    
    # Charger la calibration
    print(f"Chargement de la calibration depuis {args.calib_file}")
    calib = load_mono_calibration_file(args.calib_file)
    
    # Récupérer les images
    image_paths = get_image_files(args.input_folder)
    print(f"Images trouvées: {len(image_paths)}")
    
    if not image_paths:
        print("Erreur: Aucune image trouvée")
        sys.exit(1)
    
    # Préparer les vues
    views = prepare_views_with_mono_calibration(image_paths, calib, args.max_images)
    
    if not views:
        print("Erreur: Aucune vue valide préparée")
        sys.exit(1)
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model}...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    
    # Faire l'inférence
    print("Inférence en cours...")
    with torch.no_grad():
        predictions = model.infer(
            views,
            memory_efficient_inference=args.memory_efficient,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    
    print(f"Inférence terminée pour {len(predictions)} vues")
    
    # Créer le fichier GLB
    print(f"Génération du fichier GLB: {args.output_glb}")
    predictions_to_glb(
        predictions,
        glb_path=args.output_glb,
        cam_size=0.05,
        point_size=0.01
    )
    
    print(f"Reconstruction sauvegardée: {args.output_glb}")
    
    # Afficher quelques statistiques
    print("\n=== Statistiques ===")
    print(f"- Nombre de vues traitées: {len(predictions)}")
    
    # Calculer les statistiques de profondeur
    all_depths = []
    for i, pred in enumerate(predictions):
        depth = pred["depth_z"].cpu().numpy().squeeze()
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.extend(valid_depth.tolist())
            if i < 5:  # Afficher les stats des 5 premières vues
                print(f"- Vue {i+1}: profondeur {valid_depth.min():.2f}-{valid_depth.max():.2f}m")
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur globale: {all_depths.min():.2f}-{all_depths.max():.2f}m (moyenne: {all_depths.mean():.2f}m)")
    
    # Afficher les poses estimées
    print("\n=== Poses estimées ===")
    for i, pred in enumerate(predictions[:5]):  # 5 premières poses
        pose = pred["camera_poses"].cpu().numpy().squeeze()
        translation = pose[:3, 3]
        print(f"- Vue {i+1}: position [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")


def create_example_mono_calibration_file():
    """Crée un exemple de fichier de calibration mono-caméra."""
    example_calib = {
        'camera': {
            'intrinsics': [800.0, 800.0, 320.0, 240.0],  # [fx, fy, cx, cy]
            'distortion': [-0.1, 0.05, 0.001, 0.001, 0.0],  # [k1, k2, p1, p2, k3] - optionnel
            'image_size': [640, 480]  # [width, height]
        }
    }
    
    with open('example_mono_calib.yaml', 'w') as f:
        yaml.dump(example_calib, f, default_flow_style=False, indent=2)
    
    print("Fichier d'exemple créé: example_mono_calib.yaml")


if __name__ == "__main__":
    # Créer un fichier d'exemple si pas d'arguments
    if len(sys.argv) == 1:
        create_example_mono_calibration_file()
        print("\nUsage:")
        print("python scripts/demo_mono_calibration.py --input_folder /path/to/images --calib_file /path/to/calib.yaml")
    else:
        main()