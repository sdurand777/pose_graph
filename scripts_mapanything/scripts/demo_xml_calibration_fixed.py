#!/usr/bin/env python3
"""
Version corrigée du demo MapAnything avec calibration XML

Usage:
    # Mono
    python scripts/demo_xml_calibration_fixed.py --input_folder /path/to/images --xml_calib /path/to/calib.xml --camera_id lrl

    # Stéréo
    python scripts/demo_xml_calibration_fixed.py --left_folder /path/to/left --right_folder /path/to/right --xml_calib /path/to/calib.xml
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Configuration pour une meilleure efficacité mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.viz import predictions_to_glb
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
    
    # Parser les baselines stéréo si disponibles
    baselines = root.find('baselines')
    if baselines is not None:
        stereo = baselines.find('stereo')
        if stereo is not None:
            rotation_text = stereo.find('rotation').text.strip()
            rotation_values = [float(x) for x in rotation_text.split()]
            R = np.array(rotation_values).reshape(3, 3)
            
            translation_text = stereo.find('translation').text.strip()
            T = np.array([float(x) for x in translation_text.split()])
            
            calibration['stereo'] = {
                'R': R.tolist(),
                'T': T.tolist(),
                'baseline': np.linalg.norm(T)
            }
    
    return calibration


def scale_calibration_params(calib_params: Dict, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Dict:
    """Met à l'échelle les paramètres de calibration."""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    scaled_params = calib_params.copy()
    
    fx, fy, cx, cy = calib_params['intrinsics']
    scaled_params['intrinsics'] = [
        fx * scale_x,
        fy * scale_y,
        cx * scale_x,
        cy * scale_y
    ]
    
    scaled_params['distortion'] = calib_params['distortion'].copy()
    scaled_params['image_size'] = [target_w, target_h]
    
    print(f"Calibration mise à l'échelle: {orig_w}x{orig_h} -> {target_w}x{target_h}")
    print(f"  fx: {fx:.1f} -> {scaled_params['intrinsics'][0]:.1f}")
    print(f"  fy: {fy:.1f} -> {scaled_params['intrinsics'][1]:.1f}")
    print(f"  cx: {cx:.1f} -> {scaled_params['intrinsics'][2]:.1f}")
    print(f"  cy: {cy:.1f} -> {scaled_params['intrinsics'][3]:.1f}")
    
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


def get_image_files(folder_path: str) -> List[str]:
    """Récupère tous les fichiers d'images dans un dossier."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def process_images_with_calibration(image_paths: List[str], calib_params: Dict, max_images: Optional[int] = None) -> List[str]:
    """
    Préprocesse les images avec calibration et retourne les chemins des images corrigées.
    """
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Préparation de {len(image_paths)} images avec calibration XML")
    
    processed_paths = []
    temp_dir = "/tmp/mapanything_corrected"
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        print(f"Préparation de l'image {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur lors du chargement de l'image {i+1}, passage à la suivante")
            continue
        
        # Convertir BGR -> RGB pour le traitement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Adapter la calibration à la taille de l'image
        original_size = calib_params['image_size']
        current_size = [w, h]
        
        if current_size != original_size:
            if i == 0:  # Afficher le redimensionnement une seule fois
                print(f"Redimensionnement détecté: calibration {original_size[0]}x{original_size[1]} -> image {w}x{h}")
            scaled_params = scale_calibration_params(calib_params, original_size, current_size)
        else:
            scaled_params = calib_params
        
        # Créer la matrice d'intrinsèques et corriger la distortion
        intrinsics = create_intrinsics_matrix(scaled_params['intrinsics'])
        img_undist, K_undist = undistort_image(img_rgb, intrinsics, scaled_params['distortion'])
        
        # Sauvegarder l'image corrigée
        corrected_path = os.path.join(temp_dir, f"corrected_{i:04d}.jpg")
        cv2.imwrite(corrected_path, cv2.cvtColor(img_undist, cv2.COLOR_RGB2BGR))
        processed_paths.append(corrected_path)
    
    return processed_paths


def add_calibration_to_views(views: List[Dict], calib_params: Dict, image_paths: List[str], poses: Optional[List[np.ndarray]] = None):
    """Ajoute les informations de calibration aux vues préparées."""
    for i, (view, image_path) in enumerate(zip(views, image_paths)):
        # Charger l'image originale pour déterminer la taille
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Adapter la calibration
        original_size = calib_params['image_size']
        current_size = [w, h]
        
        if current_size != original_size:
            scaled_params = scale_calibration_params(calib_params, original_size, current_size)
        else:
            scaled_params = calib_params
        
        # Calculer les intrinsèques corrigées pour la distortion
        intrinsics = create_intrinsics_matrix(scaled_params['intrinsics'])
        dist_coeffs = np.array(scaled_params['distortion'], dtype=np.float32)
        
        new_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, dist_coeffs, (w, h), alpha=1.0
        )
        
        # Mettre à jour la vue
        view["intrinsics"] = torch.from_numpy(new_intrinsics).float()
        
        if poses is not None and i < len(poses):
            view["camera_poses"] = torch.from_numpy(poses[i]).float()
            view["is_metric_scale"] = torch.tensor([True])
        else:
            view["is_metric_scale"] = torch.tensor([False])


def main():
    parser = argparse.ArgumentParser(description="Demo MapAnything avec calibration XML (version corrigée)")
    parser.add_argument("--xml_calib", required=True, help="Fichier de calibration XML")
    parser.add_argument("--output_glb", default="xml_reconstruction.glb", help="Fichier GLB de sortie")
    parser.add_argument("--model", default="facebook/map-anything", help="Modèle MapAnything")
    parser.add_argument("--memory_efficient", action="store_true", help="Mode économe en mémoire")
    
    # Mode mono-caméra
    parser.add_argument("--input_folder", help="Dossier des images (mode mono)")
    parser.add_argument("--camera_id", default="lrl", help="ID de la caméra pour le mode mono (lrl ou lrr)")
    parser.add_argument("--max_images", type=int, help="Nombre max d'images (mode mono)")
    
    # Mode stéréo
    parser.add_argument("--left_folder", help="Dossier des images gauches (mode stéréo)")
    parser.add_argument("--right_folder", help="Dossier des images droites (mode stéréo)")
    parser.add_argument("--max_pairs", type=int, help="Nombre max de paires (mode stéréo)")
    
    args = parser.parse_args()
    
    # Vérifications
    if not os.path.exists(args.xml_calib):
        print(f"Erreur: Le fichier de calibration XML {args.xml_calib} n'existe pas")
        sys.exit(1)
    
    # Déterminer le mode
    is_stereo_mode = args.left_folder and args.right_folder
    is_mono_mode = args.input_folder
    
    if not is_stereo_mode and not is_mono_mode:
        print("Erreur: Spécifiez --input_folder (mono) OU --left_folder + --right_folder (stéréo)")
        sys.exit(1)
    
    # Charger la calibration
    print(f"Chargement de la calibration XML depuis {args.xml_calib}")
    calib = parse_xml_calibration(args.xml_calib)
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model}...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    
    if is_mono_mode:
        print(f"\n=== Mode mono-caméra (caméra {args.camera_id}) ===")
        
        if not os.path.exists(args.input_folder):
            print(f"Erreur: Le dossier {args.input_folder} n'existe pas")
            sys.exit(1)
        
        if args.camera_id not in calib:
            print(f"Erreur: Caméra {args.camera_id} non trouvée dans la calibration")
            sys.exit(1)
        
        image_paths = get_image_files(args.input_folder)
        print(f"Images trouvées: {len(image_paths)}")
        
        # Préprocesser les images avec calibration
        processed_paths = process_images_with_calibration(
            image_paths, calib[args.camera_id], args.max_images
        )
        
        # Charger les images préprocessées
        views = load_images(processed_paths)
        
        # Ajouter les informations de calibration
        add_calibration_to_views(views, calib[args.camera_id], image_paths[:len(views)])
        
    else:  # Mode stéréo
        print(f"\n=== Mode stéréo ===")
        
        if not os.path.exists(args.left_folder) or not os.path.exists(args.right_folder):
            print("Erreur: Dossiers gauche ou droite non trouvés")
            sys.exit(1)
        
        if 'stereo' not in calib:
            print("Erreur: Pas de paramètres stéréo dans la calibration XML")
            sys.exit(1)
        
        left_images = get_image_files(args.left_folder)
        right_images = get_image_files(args.right_folder)
        
        # Limiter le nombre de paires
        if args.max_pairs:
            left_images = left_images[:args.max_pairs]
            right_images = right_images[:args.max_pairs]
        
        min_count = min(len(left_images), len(right_images))
        left_images = left_images[:min_count]
        right_images = right_images[:min_count]
        
        print(f"Traitement de {min_count} paires stéréo")
        
        # Préprocesser les images
        left_processed = process_images_with_calibration(left_images, calib['lrl'])
        right_processed = process_images_with_calibration(right_images, calib['lrr'])
        
        # Charger les images
        left_views = load_images(left_processed)
        right_views = load_images(right_processed)
        
        # Créer les poses stéréo
        R = np.array(calib['stereo']['R'], dtype=np.float32)
        T = np.array(calib['stereo']['T'], dtype=np.float32)
        
        left_pose = np.eye(4, dtype=np.float32)
        right_pose = np.eye(4, dtype=np.float32)
        right_pose[:3, :3] = R
        right_pose[:3, 3] = T
        
        # Ajouter les calibrations
        left_poses = [left_pose] * len(left_views)
        right_poses = [right_pose] * len(right_views)
        
        add_calibration_to_views(left_views, calib['lrl'], left_images[:len(left_views)], left_poses)
        add_calibration_to_views(right_views, calib['lrr'], right_images[:len(right_views)], right_poses)
        
        # Combiner les vues
        views = []
        for left_view, right_view in zip(left_views, right_views):
            views.extend([left_view, right_view])
    
    if not views:
        print("Erreur: Aucune vue valide préparée")
        sys.exit(1)
    
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
    
    # Nettoyer les fichiers temporaires
    temp_dir = "/tmp/mapanything_corrected"
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    print("\n=== Terminé avec succès ! ===")


if __name__ == "__main__":
    main()