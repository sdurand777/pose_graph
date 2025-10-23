#!/usr/bin/env python3
"""
Demo MapAnything avec calibration XML (mono ou stéréo) avec redimensionnement automatique

Usage:
    # Mono (une seule caméra)
    python scripts/demo_xml_calibration.py --input_folder /path/to/images --xml_calib /path/to/calib.xml --camera_id lrl

    # Stéréo (deux caméras)
    python scripts/demo_xml_calibration.py --left_folder /path/to/left --right_folder /path/to/right --xml_calib /path/to/calib.xml
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
    """
    Parse un fichier de calibration XML et extrait les paramètres des caméras.
    
    Returns:
        Dict contenant les paramètres de calibration pour chaque caméra
    """
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
            'distortion': [k1, k2, p1, p2, k3]  # OpenCV ordre: k1, k2, p1, p2, k3
        }
    
    # Parser les baselines stéréo si disponibles
    baselines = root.find('baselines')
    if baselines is not None:
        stereo = baselines.find('stereo')
        if stereo is not None:
            # Rotation matrix (3x3)
            rotation_text = stereo.find('rotation').text.strip()
            rotation_values = [float(x) for x in rotation_text.split()]
            R = np.array(rotation_values).reshape(3, 3)
            
            # Translation vector (3x1)
            translation_text = stereo.find('translation').text.strip()
            T = np.array([float(x) for x in translation_text.split()])
            
            calibration['stereo'] = {
                'R': R.tolist(),
                'T': T.tolist(),
                'baseline': np.linalg.norm(T)
            }
    
    return calibration


def scale_calibration_params(calib_params: Dict, original_size: Tuple[int, int], target_size: Tuple[int, int]) -> Dict:
    """
    Met à l'échelle les paramètres de calibration pour une nouvelle taille d'image.
    
    Args:
        calib_params: Paramètres de calibration originaux
        original_size: Taille originale (width, height)
        target_size: Taille cible (width, height)
        
    Returns:
        Paramètres de calibration mis à l'échelle
    """
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


def undistort_image(image: np.ndarray, intrinsics: np.ndarray, distortion: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Corrige la distortion d'une image."""
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


def prepare_mono_views(image_paths: List[str], calib: Dict, camera_id: str, max_images: Optional[int] = None) -> List[Dict]:
    """Prépare les vues mono-caméra."""
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Traitement de {len(image_paths)} images avec calibration XML (caméra {camera_id})")
    
    camera_params = calib[camera_id]
    
    views = []
    first_image_processed = False
    
    for i, image_path in enumerate(image_paths):
        print(f"Préparation de l'image {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur lors du chargement de l'image {i+1}, passage à la suivante")
            continue
        
        # Convertir BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Adapter la calibration à la taille de l'image actuelle si nécessaire
        original_size = camera_params['image_size']
        current_size = [w, h]
        
        if current_size != original_size:
            if not first_image_processed:
                print(f"Redimensionnement détecté: calibration {original_size[0]}x{original_size[1]} -> image {w}x{h}")
            scaled_params = scale_calibration_params(camera_params, original_size, current_size)
        else:
            scaled_params = camera_params
        
        # Créer la matrice d'intrinsèques
        intrinsics = create_intrinsics_matrix(scaled_params['intrinsics'])
        
        # Corriger la distortion
        img_undist, K_undist = undistort_image(img, intrinsics, scaled_params['distortion'])
        
        # Sauvegarder temporairement l'image pour utiliser load_images
        temp_path = f"/tmp/temp_img_{i}.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img_undist, cv2.COLOR_RGB2BGR))
        
        # Utiliser load_images pour le preprocessing correct
        temp_views = load_images([temp_path])
        if not temp_views:
            print(f"Erreur lors du preprocessing de l'image {i+1}, passage à la suivante")
            os.remove(temp_path)
            continue
        
        # Créer la vue avec les bonnes données normalisées
        view = temp_views[0].copy()
        view["intrinsics"] = torch.from_numpy(K_undist).float()
        view["is_metric_scale"] = torch.tensor([False])
        view["idx"] = torch.tensor([i])
        
        # Nettoyer le fichier temporaire
        os.remove(temp_path)
        
        views.append(view)
        first_image_processed = True
    
    return views


def prepare_stereo_views(left_images: List[str], right_images: List[str], calib: Dict, max_pairs: Optional[int] = None) -> List[Dict]:
    """Prépare les vues stéréo."""
    if max_pairs:
        left_images = left_images[:max_pairs]
        right_images = right_images[:max_pairs]
    
    min_count = min(len(left_images), len(right_images))
    left_images = left_images[:min_count]
    right_images = right_images[:min_count]
    
    print(f"Traitement de {min_count} paires stéréo avec calibration XML")
    
    # Extraire les paramètres de calibration
    left_params = calib['lrl']
    right_params = calib['lrr']
    stereo_params = calib['stereo']
    
    # Créer les poses stéréo
    R = np.array(stereo_params['R'], dtype=np.float32)
    T = np.array(stereo_params['T'], dtype=np.float32)
    
    left_pose = np.eye(4, dtype=np.float32)
    right_pose = np.eye(4, dtype=np.float32)
    right_pose[:3, :3] = R
    right_pose[:3, 3] = T
    
    print(f"Baseline stéréo: {stereo_params['baseline']:.3f}m")
    
    views = []
    first_pair_processed = False
    
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
        
        # Adapter la calibration si nécessaire
        h, w = left_img.shape[:2]
        current_size = [w, h]
        
        # Left camera
        original_size_left = left_params['image_size']
        if current_size != original_size_left:
            if not first_pair_processed:
                print(f"Redimensionnement gauche: {original_size_left[0]}x{original_size_left[1]} -> {w}x{h}")
            left_scaled = scale_calibration_params(left_params, original_size_left, current_size)
        else:
            left_scaled = left_params
        
        # Right camera
        h_r, w_r = right_img.shape[:2]
        current_size_right = [w_r, h_r]
        original_size_right = right_params['image_size']
        if current_size_right != original_size_right:
            if not first_pair_processed:
                print(f"Redimensionnement droite: {original_size_right[0]}x{original_size_right[1]} -> {w_r}x{h_r}")
            right_scaled = scale_calibration_params(right_params, original_size_right, current_size_right)
        else:
            right_scaled = right_params
        
        # Traiter les images
        left_intrinsics = create_intrinsics_matrix(left_scaled['intrinsics'])
        right_intrinsics = create_intrinsics_matrix(right_scaled['intrinsics'])
        
        left_img_undist, left_K_undist = undistort_image(left_img, left_intrinsics, left_scaled['distortion'])
        right_img_undist, right_K_undist = undistort_image(right_img, right_intrinsics, right_scaled['distortion'])
        
        # Sauvegarder temporairement les images pour utiliser load_images
        temp_left_path = f"/tmp/temp_left_{i}.jpg"
        temp_right_path = f"/tmp/temp_right_{i}.jpg"
        cv2.imwrite(temp_left_path, cv2.cvtColor(left_img_undist, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_right_path, cv2.cvtColor(right_img_undist, cv2.COLOR_RGB2BGR))
        
        # Utiliser load_images pour le preprocessing correct
        temp_left_views = load_images([temp_left_path])
        temp_right_views = load_images([temp_right_path])
        
        if not temp_left_views or not temp_right_views:
            print(f"Erreur lors du preprocessing de la paire {i+1}, passage à la suivante")
            for temp_path in [temp_left_path, temp_right_path]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            continue
        
        # Créer les vues avec les bonnes données normalisées
        left_view = temp_left_views[0].copy()
        left_view["intrinsics"] = torch.from_numpy(left_K_undist).float()
        left_view["camera_poses"] = torch.from_numpy(left_pose).float()
        left_view["is_metric_scale"] = torch.tensor([True])
        left_view["idx"] = torch.tensor([i * 2])
        
        right_view = temp_right_views[0].copy()
        right_view["intrinsics"] = torch.from_numpy(right_K_undist).float()
        right_view["camera_poses"] = torch.from_numpy(right_pose).float()
        right_view["is_metric_scale"] = torch.tensor([True])
        right_view["idx"] = torch.tensor([i * 2 + 1])
        
        # Nettoyer les fichiers temporaires
        os.remove(temp_left_path)
        os.remove(temp_right_path)
        
        views.extend([left_view, right_view])
        first_pair_processed = True
    
    return views


def main():
    parser = argparse.ArgumentParser(description="Demo MapAnything avec calibration XML")
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
    
    # Déterminer le mode (mono ou stéréo)
    is_stereo_mode = args.left_folder and args.right_folder
    is_mono_mode = args.input_folder
    
    if not is_stereo_mode and not is_mono_mode:
        print("Erreur: Spécifiez --input_folder (mono) OU --left_folder + --right_folder (stéréo)")
        sys.exit(1)
    
    if is_stereo_mode and is_mono_mode:
        print("Erreur: Choisissez entre mode mono (--input_folder) OU stéréo (--left_folder + --right_folder)")
        sys.exit(1)
    
    # Charger la calibration
    print(f"Chargement de la calibration XML depuis {args.xml_calib}")
    calib = parse_xml_calibration(args.xml_calib)
    
    # Afficher les infos de calibration
    print("\n=== Calibration XML chargée ===")
    for camera_id, params in calib.items():
        if camera_id != 'stereo':
            print(f"Caméra {camera_id}:")
            print(f"  - Taille originale: {params['image_size'][0]}x{params['image_size'][1]}")
            print(f"  - Intrinsèques: fx={params['intrinsics'][0]:.1f}, fy={params['intrinsics'][1]:.1f}")
            print(f"  - Centre: cx={params['intrinsics'][2]:.1f}, cy={params['intrinsics'][3]:.1f}")
            print(f"  - Distortion: k1={params['distortion'][0]:.4f}, k2={params['distortion'][1]:.4f}")
    
    if 'stereo' in calib:
        print(f"Stéréo - Baseline: {calib['stereo']['baseline']:.3f}m")
    
    # Préparer les vues selon le mode
    if is_mono_mode:
        print(f"\n=== Mode mono-caméra (caméra {args.camera_id}) ===")
        
        if not os.path.exists(args.input_folder):
            print(f"Erreur: Le dossier {args.input_folder} n'existe pas")
            sys.exit(1)
        
        if args.camera_id not in calib:
            print(f"Erreur: Caméra {args.camera_id} non trouvée dans la calibration")
            print(f"Caméras disponibles: {list(calib.keys())}")
            sys.exit(1)
        
        image_paths = get_image_files(args.input_folder)
        print(f"Images trouvées: {len(image_paths)}")
        
        views = prepare_mono_views(image_paths, calib, args.camera_id, args.max_images)
        
    else:  # Mode stéréo
        print(f"\n=== Mode stéréo ===")
        
        if not os.path.exists(args.left_folder):
            print(f"Erreur: Le dossier gauche {args.left_folder} n'existe pas")
            sys.exit(1)
        
        if not os.path.exists(args.right_folder):
            print(f"Erreur: Le dossier droite {args.right_folder} n'existe pas")
            sys.exit(1)
        
        if 'stereo' not in calib:
            print("Erreur: Pas de paramètres stéréo dans la calibration XML")
            sys.exit(1)
        
        left_images = get_image_files(args.left_folder)
        right_images = get_image_files(args.right_folder)
        print(f"Images trouvées - Gauche: {len(left_images)}, Droite: {len(right_images)}")
        
        views = prepare_stereo_views(left_images, right_images, calib, args.max_pairs)
    
    if not views:
        print("Erreur: Aucune vue valide préparée")
        sys.exit(1)
    
    # Configuration du device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUtilisation du device: {device}")
    
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
    if is_stereo_mode:
        print(f"- Nombre de vues traitées: {len(predictions)} ({len(predictions)//2} paires stéréo)")
    else:
        print(f"- Nombre de vues traitées: {len(predictions)}")
    
    # Calculer les statistiques de profondeur
    all_depths = []
    for i, pred in enumerate(predictions):
        depth = pred["depth_z"].cpu().numpy().squeeze()
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0:
            all_depths.extend(valid_depth.tolist())
    
    if all_depths:
        all_depths = np.array(all_depths)
        print(f"- Profondeur globale: {all_depths.min():.2f}-{all_depths.max():.2f}m (moyenne: {all_depths.mean():.2f}m)")


if __name__ == "__main__":
    main()