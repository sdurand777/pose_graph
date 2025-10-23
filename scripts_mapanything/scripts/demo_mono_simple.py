#!/usr/bin/env python3
"""
Demo MapAnything simple en mode mono avec pré-correction de la distortion XML

Usage:
    python scripts/demo_mono_simple.py --input_folder /path/to/images --xml_calib /path/to/calib.xml --camera_id lrl --output_glb output.glb
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

# Configuration pour une meilleure efficacité mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.hf_utils.viz import predictions_to_glb


def parse_xml_calibration(xml_path: str) -> Dict:
    """Parse un fichier de calibration XML et extrait les paramètres."""
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
            'distortion': [k1, k2, p1, p2, k3]  # OpenCV ordre
        }
    
    return calibration


def scale_calibration_params(calib_params: Dict, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Dict:
    """Met à l'échelle les paramètres de calibration pour une nouvelle taille d'image."""
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    scaled_params = calib_params.copy()
    
    # Mettre à l'échelle les intrinsèques
    fx, fy, cx, cy = calib_params['intrinsics']
    scaled_params['intrinsics'] = [
        fx * scale_x,  # fx
        fy * scale_y,  # fy  
        cx * scale_x,  # cx
        cy * scale_y   # cy
    ]
    
    # Les coefficients de distortion ne changent pas
    scaled_params['distortion'] = calib_params['distortion'].copy()
    scaled_params['image_size'] = [target_w, target_h]
    
    print(f"Calibration mise à l'échelle: {orig_w}x{orig_h} -> {target_w}x{target_h}")
    print(f"  fx: {fx:.1f} -> {scaled_params['intrinsics'][0]:.1f}")
    print(f"  fy: {fy:.1f} -> {scaled_params['intrinsics'][1]:.1f}")
    print(f"  cx: {cx:.1f} -> {scaled_params['intrinsics'][2]:.1f}")
    print(f"  cy: {cy:.1f} -> {scaled_params['intrinsics'][3]:.1f}")
    
    return scaled_params


def create_intrinsics_matrix(intrinsics_list: List[float]) -> np.ndarray:
    """Crée une matrice d'intrinsèques 3x3 depuis une liste [fx, fy, cx, cy]."""
    fx, fy, cx, cy = intrinsics_list
    return np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=np.float32)


def undistort_image_and_save(image_path: str, output_path: str, calib_params: Dict) -> bool:
    """
    Corrige la distortion d'une image et la sauvegarde.
    
    Returns:
        True si réussi, False sinon
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # Adapter la calibration à la taille de l'image actuelle
    original_size = calib_params['image_size']
    current_size = [w, h]
    
    if current_size != original_size:
        scaled_params = scale_calibration_params(calib_params, original_size, current_size)
    else:
        scaled_params = calib_params
    
    # Créer la matrice d'intrinsèques
    intrinsics = create_intrinsics_matrix(scaled_params['intrinsics'])
    
    # Coefficients de distortion
    dist_coeffs = np.array(scaled_params['distortion'], dtype=np.float32)
    
    # Corriger la distortion
    img_undistorted = cv2.undistort(img, intrinsics, dist_coeffs)
    
    # Sauvegarder l'image corrigée
    cv2.imwrite(output_path, img_undistorted)
    
    return True


def get_image_files(folder_path: str) -> List[str]:
    """Récupère tous les fichiers d'images dans un dossier."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files


def main():
    parser = argparse.ArgumentParser(description="Demo MapAnything mono simple avec calibration XML")
    parser.add_argument("--input_folder", required=True, help="Dossier contenant les images")
    parser.add_argument("--xml_calib", required=True, help="Fichier de calibration XML")
    parser.add_argument("--camera_id", required=True, choices=['lrl', 'lrr'], help="ID de la caméra (lrl ou lrr)")
    parser.add_argument("--output_glb", required=True, help="Fichier GLB de sortie")
    parser.add_argument("--model", default="facebook/map-anything", help="Modèle MapAnything")
    parser.add_argument("--max_images", type=int, help="Nombre maximum d'images à traiter")
    parser.add_argument("--memory_efficient", action="store_true", help="Mode économe en mémoire")
    
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
    print(f"\nCalibration {args.camera_id} chargée:")
    print(f"  - Taille originale: {camera_params['image_size'][0]}x{camera_params['image_size'][1]}")
    print(f"  - Intrinsèques: fx={camera_params['intrinsics'][0]:.1f}, fy={camera_params['intrinsics'][1]:.1f}")
    print(f"  - Centre: cx={camera_params['intrinsics'][2]:.1f}, cy={camera_params['intrinsics'][3]:.1f}")
    print(f"  - Distortion: k1={camera_params['distortion'][0]:.4f}, k2={camera_params['distortion'][1]:.4f}")
    
    # Récupérer les images
    image_files = get_image_files(args.input_folder)
    if not image_files:
        print(f"Erreur: Aucune image trouvée dans {args.input_folder}")
        sys.exit(1)
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nTraitement de {len(image_files)} images...")
    
    # Créer un dossier temporaire pour les images corrigées
    temp_dir = "/tmp/mapanything_undistorted"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Corriger la distortion de toutes les images
    corrected_images = []
    for i, image_path in enumerate(image_files):
        print(f"Correction de l'image {i+1}/{len(image_files)}: {Path(image_path).name}")
        
        output_path = os.path.join(temp_dir, f"undistorted_{i:04d}.jpg")
        
        if undistort_image_and_save(image_path, output_path, camera_params):
            corrected_images.append(output_path)
        else:
            print(f"Erreur lors de la correction de l'image {i+1}")
    
    if not corrected_images:
        print("Erreur: Aucune image corrigée avec succès")
        sys.exit(1)
    
    print(f"\n{len(corrected_images)} images corrigées avec succès")
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device: {device}")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model}...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    
    # Charger les images corrigées avec load_images (preprocessing standard)
    print("Chargement des images corrigées...")
    views = load_images(corrected_images)
    
    if not views:
        print("Erreur: Impossible de charger les images corrigées")
        sys.exit(1)
    
    print(f"Images chargées: {len(views)}")
    
    # Faire l'inférence (mode standard - MapAnything estime les poses)
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
    
    # Préparer les données pour l'export GLB
    print(f"Préparation des données pour GLB...")
    world_points_list = []
    images_list = []
    masks_list = []
    
    for i, pred in enumerate(predictions):
        # Extraire les points 3D dans le repère mondial
        pts3d = pred["pts3d"].cpu().numpy()  # Shape: (H, W, 3)
        
        # Extraire l'image dénormalisée
        image_np = pred["img_no_norm"][0].cpu().numpy()  # Shape: (H, W, 3)
        
        # Créer un masque de validité
        mask = pred["mask"][0].cpu().numpy().squeeze()  # Shape: (H, W)
        if mask.ndim == 3:
            mask = mask.any(axis=-1)  # Réduire à 2D si nécessaire
        
        # Ajouter aux listes
        world_points_list.append(pts3d)
        images_list.append(image_np)
        masks_list.append(mask)
    
    # Empiler toutes les vues
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    final_masks = np.stack(masks_list, axis=0)
    
    # Créer le dictionnaire de prédictions pour l'export GLB
    predictions_dict = {
        "world_points": world_points,
        "images": images,
        "final_mask": final_masks,  # Utiliser 'final_mask' (singulier)
        "extrinsic": np.stack([pred["camera_poses"].cpu().numpy() for pred in predictions], axis=0)
    }
    
    # Créer le fichier GLB
    print(f"Génération du fichier GLB: {args.output_glb}")
    scene = predictions_to_glb(
        predictions_dict,
        show_cam=False,  # Pas d'affichage des caméras (comme en mode normal)
        as_mesh=False,   # Utiliser des nuages de points
    )
    
    # Sauvegarder le fichier GLB
    scene.export(args.output_glb)
    
    print(f"Reconstruction sauvegardée: {args.output_glb}")
    
    # Nettoyer les fichiers temporaires
    print("Nettoyage des fichiers temporaires...")
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Afficher les statistiques
    print("\n=== Statistiques ===")
    print(f"- Images traitées: {len(predictions)}")
    print(f"- Calibration utilisée: {args.camera_id}")
    print(f"- Correction de distortion: Appliquée")
    
    # Calculer les statistiques de profondeur
    all_depths = []
    for pred in predictions:
        depth = pred["depth_z"].cpu().numpy().squeeze()
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.extend(valid_depth.tolist())
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur globale: {all_depths.min():.2f}-{all_depths.max():.2f}m (moyenne: {all_depths.mean():.2f}m)")
    
    print("\n=== Terminé avec succès ! ===")
    print(f"Pour visualiser: Blender > Import > glTF 2.0 > {args.output_glb}")


if __name__ == "__main__":
    main()