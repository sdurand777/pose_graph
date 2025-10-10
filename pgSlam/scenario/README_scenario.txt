# Fichiers de scénario - Poses réelles avec Loop Closures
===========================================================

## Fichiers source

1. **id_kf_2_idx_img.txt**
   - Liste Python des IDs d'images correspondant aux keyframes
   - Format: [10, 11, 13, 14, ...]
   - 1198 keyframes

2. **vertices_stan.txt**  
   - Poses 3D pour chaque image ID
   - Format CSV: id,tx,ty,tz,qx,qy,qz,qw
   - 1195 poses (certaines images n'ont pas de pose)

3. **loop_all.txt**
   - Paires de loop closures entre images
   - Format: img00971 img00059
   - 95 lignes (mais seulement 7 loop closures valides après filtrage)

## Fichier généré

**poses_with_loops.txt** (généré par generate_poses_with_loops.py)
- Toutes les poses avec leurs loop closures associées
- Format CSV: kf_idx,img_idx,tx,ty,tz,qx,qy,qz,qw,loop_ids

Colonnes:
- kf_idx: Index du keyframe (0 à 1197)
- img_idx: ID de l'image originale
- tx,ty,tz: Position 3D en mètres
- qx,qy,qz,qw: Orientation quaternion
- loop_ids: Liste des kf_idx en loop closure (séparés par ';')
  - Vide si pas de loop closure
  - Exemple: "313" ou "417;419"

## Statistiques

- Total keyframes: 1198
- Keyframes avec poses: ~390 (beaucoup d'images n'ont pas de pose)
- Loop closures détectées: 7 paires
- Keyframes impliqués dans loop closures: 14

## Utilisation

### Générer le fichier des poses
```bash
python generate_poses_with_loops.py
```

### Visualiser la trajectoire
```bash
python visualize_scenario.py
```

Cela génère 4 vues de la trajectoire:
- Vue 3D interactive
- Vue de dessus (XY)
- Vue de côté (XZ)
- Vue de face (YZ)

Et sauvegarde l'image dans: scenario/trajectory_visualization.png

## Format des données

### Exemple de ligne SANS loop closure:
```
0,10,-0.352441460,1.025669813,-0.909772396,0.025656046,-0.213963419,-0.182734922,0.959256768,
```

### Exemple de ligne AVEC loop closure:
```
18,59,1.886414528,4.294839382,-8.265888214,-0.164887652,-0.276930213,-0.163531467,0.932408452,313
```
Cette pose (kf_idx=18) a une loop closure avec la pose kf_idx=313.

### Exemple avec PLUSIEURS loop closures:
```
128,412,10.049818993,3.388557673,-11.530387878,0.286152810,0.454635918,0.516600311,-0.666769862,417;419
```
Cette pose a des loop closures avec les poses 417 et 419.

## Notes importantes

1. **Poses manquantes**: Beaucoup d'images ID dans id_kf_2_idx_img n'ont pas de pose correspondante 
   dans vertices_stan.txt (IDs > 1194). Ces lignes sont quand même présentes dans le fichier de 
   sortie mais avec des valeurs par défaut ou vides.

2. **Loop closures filtrées**: Sur les 95 lignes de loop_all.txt, seulement 7 sont valides car:
   - Les deux images doivent être dans id_kf_2_idx_img
   - Les deux images doivent avoir une pose dans vertices_stan.txt

3. **Système de coordonnées**: 
   - Position en mètres (x, y, z)
   - Orientation en quaternions (x, y, z, w) normalisés

4. **Quaternions**: Format (qx, qy, qz, qw) compatible avec scipy.spatial.transform.Rotation

## Conversion en G2O

Pour utiliser ces données avec g2o, vous pouvez créer un fichier .g2o:

```python
# Exemple de conversion (voir g2o_io_3d.py)
VERTEX_SE3:QUAT <kf_idx> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
EDGE_SE3:QUAT <kf_idx1> <kf_idx2> <dx> <dy> <dz> <dqx> <dqy> <dqz> <dqw> <info_matrix>
```

## Intégration avec le système 3D

Les données de ce scénario peuvent être utilisées avec les modules 3D:
- trajectory_3d.py: Pour comparer avec des trajectoires synthétiques
- g2o_io_3d.py: Pour écrire au format G2O
- visualization_3d.py: Pour visualiser
- metrics_3d.py: Pour calculer des métriques d'erreur

