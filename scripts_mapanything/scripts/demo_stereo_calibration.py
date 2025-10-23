#!/usr/bin/env python3
"""
Demo MapAnything avec calibration complète (intrinsèques + distortion + extrinsèques stéréo)

Usage:
    python scripts/demo_stereo_calibration.py --left_folder /path/to/left --right_folder /path/to/right --calib_file /path/to/calib.yaml
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


def load_calibration_file(calib_path: str) -> Dict:
    """
    Charge un fichier de calibration YAML.
    
    Format attendu:
    left_camera:
      intrinsics: [fx, fy, cx, cy]
      distortion: [k1, k2, p1, p2, k3]  # Optionnel
      image_size: [width, height]
    right_camera:
      intrinsics: [fx, fy, cx, cy]
      distortion: [k1, k2, p1, p2, k3]  # Optionnel
      image_size: [width, height]
    stereo:
      R: [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]  # Rotation entre caméras
      T: [tx, ty, tz]  # Translation entre caméras
      baseline: 0.12  # Distance entre caméras (optionnel)
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


def create_stereo_camera_poses(stereo_calib: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée les poses des caméras gauche et droite.
    
    Args:
        stereo_calib: Calibration stéréo avec R et T
        
    Returns:
        left_pose: Pose de la caméra gauche (4, 4) - identité par convention
        right_pose: Pose de la caméra droite (4, 4) - relative à gauche
    """
    # Caméra gauche = référence (identité)
    left_pose = np.eye(4, dtype=np.float32)
    
    # Caméra droite = transformation depuis gauche
    R = np.array(stereo_calib['R'], dtype=np.float32)
    T = np.array(stereo_calib['T'], dtype=np.float32)
    
    right_pose = np.eye(4, dtype=np.float32)
    right_pose[:3, :3] = R
    right_pose[:3, 3] = T
    
    return left_pose, right_pose


def get_image_files(folder_path: str) -> List[str]:
    """Récupère tous les fichiers d'images dans un dossier."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def prepare_views_with_calibration(
    left_images: List[str], 
    right_images: List[str], 
    calib: Dict,
    max_pairs: Optional[int] = None
) -> List[Dict]:
    """
    Prépare les vues avec calibration complète pour MapAnything.
    
    Args:
        left_images: Liste des chemins des images gauches
        right_images: Liste des chemins des images droites
        calib: Dictionnaire de calibration
        max_pairs: Nombre maximum de paires à traiter
        
    Returns:
        Liste des vues préparées pour MapAnything
    """
    # Limiter le nombre d'images si spécifié
    if max_pairs:
        left_images = left_images[:max_pairs]
        right_images = right_images[:max_pairs]
    
    # Vérifier que nous avons le même nombre d'images
    min_count = min(len(left_images), len(right_images))
    left_images = left_images[:min_count]
    right_images = right_images[:min_count]
    
    print(f"Traitement de {min_count} paires stéréo")
    
    # Extraire les paramètres de calibration
    left_intrinsics = create_intrinsics_matrix(calib['left_camera']['intrinsics'])
    right_intrinsics = create_intrinsics_matrix(calib['right_camera']['intrinsics'])
    
    left_distortion = calib['left_camera'].get('distortion', None)
    right_distortion = calib['right_camera'].get('distortion', None)
    
    # Créer les poses stéréo
    left_pose, right_pose = create_stereo_camera_poses(calib['stereo'])
    
    views = []
    
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"Préparation de la paire {i+1}/{min_count}")
        
        # Charger les images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print(f"Erreur lors du chargement de la paire {i+1}, passage à la suivante")
            continue
        
        # Convertir BGR -> RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Corriger la distortion si nécessaire
        left_img_undist, left_K_undist = undistort_image(left_img, left_intrinsics, left_distortion)
        right_img_undist, right_K_undist = undistort_image(right_img, right_intrinsics, right_distortion)
        
        # Créer les vues pour MapAnything
        left_view = {
            "img": torch.from_numpy(left_img_undist).float(),
            "intrinsics": torch.from_numpy(left_K_undist).float(),
            "camera_poses": torch.from_numpy(left_pose).float(),
            "is_metric_scale": torch.tensor([True]),
            "idx": torch.tensor([i * 2]),  # Index pairs
        }
        
        right_view = {
            "img": torch.from_numpy(right_img_undist).float(),
            "intrinsics": torch.from_numpy(right_K_undist).float(),
            "camera_poses": torch.from_numpy(right_pose).float(),
            "is_metric_scale": torch.tensor([True]),
            "idx": torch.tensor([i * 2 + 1]),  # Index impairs
        }
        
        views.extend([left_view, right_view])
    
    return views


def main():
    parser = argparse.ArgumentParser(description="Demo MapAnything avec calibration stéréo complète")
    parser.add_argument("--left_folder", required=True, help="Dossier des images de la caméra gauche")
    parser.add_argument("--right_folder", required=True, help="Dossier des images de la caméra droite")
    parser.add_argument("--calib_file", required=True, help="Fichier de calibration YAML")
    parser.add_argument("--output_glb", default="stereo_reconstruction.glb", help="Fichier GLB de sortie")
    parser.add_argument("--model", default="facebook/map-anything", 
                       help="Modèle à utiliser (facebook/map-anything ou facebook/map-anything-apache)")
    parser.add_argument("--max_pairs", type=int, default=None, help="Nombre maximum de paires à traiter")
    parser.add_argument("--memory_efficient", action="store_true", help="Mode économe en mémoire")
    
    args = parser.parse_args()
    
    # Vérifications
    if not os.path.exists(args.left_folder):
        print(f"Erreur: Le dossier gauche {args.left_folder} n'existe pas")
        sys.exit(1)
    
    if not os.path.exists(args.right_folder):
        print(f"Erreur: Le dossier droite {args.right_folder} n'existe pas")
        sys.exit(1)
    
    if not os.path.exists(args.calib_file):
        print(f"Erreur: Le fichier de calibration {args.calib_file} n'existe pas")
        sys.exit(1)
    
    # Charger la calibration
    print(f"Chargement de la calibration depuis {args.calib_file}")
    calib = load_calibration_file(args.calib_file)
    
    # Afficher les infos de calibration
    print("=== Calibration chargée ===")
    print(f"Caméra gauche - Intrinsèques: {calib['left_camera']['intrinsics']}")
    if 'distortion' in calib['left_camera']:
        print(f"Caméra gauche - Distortion: {calib['left_camera']['distortion']}")
    print(f"Caméra droite - Intrinsèques: {calib['right_camera']['intrinsics']}")
    if 'distortion' in calib['right_camera']:
        print(f"Caméra droite - Distortion: {calib['right_camera']['distortion']}")
    print(f"Stéréo - Baseline: {np.linalg.norm(calib['stereo']['T']):.3f}m")
    
    # Récupérer les images
    left_images = get_image_files(args.left_folder)
    right_images = get_image_files(args.right_folder)
    
    print(f"Images trouvées - Gauche: {len(left_images)}, Droite: {len(right_images)}")
    
    # Préparer les vues
    views = prepare_views_with_calibration(left_images, right_images, calib, args.max_pairs)
    
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
    print(f"- Nombre de paires stéréo: {len(predictions) // 2}")
    
    # Calculer les statistiques de profondeur pour chaque vue
    all_depths = []
    for i, pred in enumerate(predictions):
        depth = pred["depth_z"].cpu().numpy().squeeze()
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.extend(valid_depth.tolist())
            if i < 4:  # Afficher les stats des 4 premières vues
                camera_type = "Gauche" if i % 2 == 0 else "Droite"
                pair_num = i // 2 + 1
                print(f"- Paire {pair_num} ({camera_type}): profondeur {valid_depth.min():.2f}-{valid_depth.max():.2f}m")
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur globale: {all_depths.min():.2f}-{all_depths.max():.2f}m (moyenne: {all_depths.mean():.2f}m)")


def create_example_calibration_file():
    """Crée un exemple de fichier de calibration."""
    example_calib = {
        'left_camera': {
            'intrinsics': [800.0, 800.0, 320.0, 240.0],  # [fx, fy, cx, cy]
            'distortion': [-0.1, 0.05, 0.001, 0.001, 0.0],  # [k1, k2, p1, p2, k3]
            'image_size': [640, 480]  # [width, height]
        },
        'right_camera': {
            'intrinsics': [800.0, 800.0, 320.0, 240.0],
            'distortion': [-0.1, 0.05, 0.001, 0.001, 0.0],
            'image_size': [640, 480]
        },
        'stereo': {
            'R': [  # Matrice de rotation 3x3 (right relative to left)
                [0.999, -0.001, 0.002],
                [0.001, 0.999, -0.001],
                [-0.002, 0.001, 0.999]
            ],
            'T': [-0.12, 0.0, 0.0],  # Translation [tx, ty, tz] en mètres
            'baseline': 0.12  # Distance entre caméras (optionnel)
        }
    }
    
    with open('example_stereo_calib.yaml', 'w') as f:
        yaml.dump(example_calib, f, default_flow_style=False, indent=2)
    
    print("Fichier d'exemple créé: example_stereo_calib.yaml")


if __name__ == "__main__":
    # Créer un fichier d'exemple si pas d'arguments
    if len(sys.argv) == 1:
        create_example_calibration_file()
        print("\nUsage:")
        print("python scripts/demo_stereo_calibration.py --left_folder /path/to/left --right_folder /path/to/right --calib_file /path/to/calib.yaml")
    else:
        main()