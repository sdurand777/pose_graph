# 3D Pose Graph SLAM

Version 3D du système de démonstration de SLAM par graphe de poses. Cette implémentation étend le système 2D pour gérer des trajectoires complètes en 3D avec des poses SE(3).

## Différences clés par rapport à la version 2D

### Représentation des poses
- **2D (SE2)**: `(x, y, θ)` - position 2D + angle de rotation
- **3D (SE3)**: `(x, y, z, qx, qy, qz, qw)` - position 3D + quaternion pour la rotation

### Format G2O
- **2D**: `VERTEX_SE2` et `EDGE_SE2`
- **3D**: `VERTEX_SE3:QUAT` et `EDGE_SE3:QUAT`

### Matrices d'information
- **2D**: 3×3 (x, y, θ) → 6 valeurs triangulaires supérieures
- **3D**: 6×6 (x, y, z, roll, pitch, yaw) → 21 valeurs triangulaires supérieures

## Installation

### Dépendances Python
```bash
pip install numpy scipy matplotlib
```

### Optimiseur G2O
Le système nécessite g2o installé avec support SE3:
```bash
# Vérifier l'installation
which g2o

# G2O doit supporter VERTEX_SE3:QUAT et EDGE_SE3:QUAT
```

## Utilisation

### Exécution basique
```bash
python main_3d.py
```

### Paramètres disponibles

**Trajectoire & Optimisation:**
- `--trajectory-type <type>`: Type de trajectoire (`uturn` ou `helix`, défaut: `uturn`)
- `--num-poses <value>`: Nombre de poses par segment (défaut: 20)
- `--odom-info <value>`: Valeur de la matrice d'information pour l'odométrie (défaut: 500.0)
- `--loop-info <value>`: Valeur de la matrice d'information pour les loop closures (défaut: 700.0)

**Loop Closures:**
- `--loop-sampling <value>`: Échantillonner chaque Nième pose (défaut: 2). Plus élevé = moins de boucles. 0 = désactiver
- `--loop-from-noisy`: Utiliser les poses bruitées au lieu de la vérité terrain pour les loop closures
- `--loop-noise-sigma <value>`: Écart-type du bruit gaussien pour les observations de loop closure (défaut: 0.0)

**Bruit & Reproductibilité:**
- `--noise-sigma <value>`: Écart-type du bruit gaussien pour l'odométrie (défaut: 0.03)
- `--seed <value>`: Graine aléatoire pour la génération du bruit (défaut: None)

**Visualisation:**
- `--no-viz`: Désactiver la visualisation (exécution plus rapide)

## Exemples d'utilisation

### Trajectoire en U-turns 3D (défaut)
```bash
python main_3d.py --trajectory-type uturn --seed 42
```

### Trajectoire hélicoïdale
```bash
python main_3d.py --trajectory-type helix --seed 42
```

### Plus de poses avec loop closures denses
```bash
python main_3d.py --num-poses 30 --loop-sampling 1
```

### Bruit d'odométrie élevé
```bash
python main_3d.py --noise-sigma 0.1 --seed 42
```

### Loop closures depuis poses bruitées (scénario réaliste)
```bash
python main_3d.py --loop-from-noisy --loop-noise-sigma 0.05 --seed 42
```

### Configuration complète
```bash
python main_3d.py --trajectory-type helix --num-poses 25 --loop-sampling 2 \
  --noise-sigma 0.05 --loop-noise-sigma 0.02 --seed 123 \
  --odom-info 500.0 --loop-info 700.0
```

## Architecture des modules 3D

### trajectory_3d.py
Génération de trajectoires synthétiques en 3D:
- `genTraj3D()`: Trajectoire avec U-turns et variations d'altitude
- `genTrajSpiralHelix3D()`: Trajectoire hélicoïdale ascendante
- `addNoise3D()`: Ajoute du bruit gaussien aux transformations SE(3)
- `quaternion_from_euler()` / `euler_from_quaternion()`: Conversions de représentation

### g2o_io_3d.py
Gestion des fichiers G2O pour SE(3):
- `writeOdom3D()`: Crée les entrées `VERTEX_SE3:QUAT` et `EDGE_SE3:QUAT` pour l'odométrie
- `writeLoop3D()`: Ajoute les entrées `EDGE_SE3:QUAT` pour les loop closures
- `readG2o3D()`: Parse les positions optimisées depuis le fichier g2o de sortie

### visualization_3d.py
Visualisation 3D avec Matplotlib:
- `draw3D()`: Trajectoire unique avec flèches d'orientation 3D
- `drawTwo3D()`: Comparaison vérité terrain vs optimisée
- `drawThree3D()`: Affiche vérité terrain, optimisée et bruitée
- `plotErrors3D()`: Graphiques d'erreurs par pose
- `set_axes_equal()`: Assure un ratio d'aspect égal pour les axes 3D

### metrics_3d.py
Évaluation des erreurs de trajectoire 3D:
- `compute_pose_errors_3d()`: Calcule les erreurs de position (distance euclidienne 3D) et d'orientation (distance géodésique sur SO(3))
- `print_metrics_comparison_3d()`: Affichage formaté des métriques

### pose_graph_3d.py
Wrapper pour l'optimisation:
- `optimize3D()`: Exécute l'optimiseur g2o en ligne de commande

## Transformations SE(3)

Le système utilise des transformations SE(3) représentées par des matrices homogènes 4×4:

```
T = [R  t]
    [0  1]

où R est une matrice de rotation 3×3 (depuis quaternion)
et t est le vecteur de translation 3D
```

Les transformations relatives entre poses sont calculées comme: `T2_1 = inv(T1_w) @ T2_w`

## Types de trajectoires

### U-Turn 3D (`--trajectory-type uturn`)
- 6 segments avec 3 passages avant et 3 U-turns
- Variations d'altitude (montées, descentes)
- Roulis (roll) et tangage (pitch) dans les virages
- Loop closures naturelles entre trajectoires parallèles

### Helix Spiral (`--trajectory-type helix`)
- Spirale hélicoïdale ascendante
- Rayon croissant avec l'altitude
- Nombreuses opportunités de loop closures en 3D
- Excellente pour tester la robustesse avec des trajectoires complexes

## Détection de loop closures 3D

Les loop closures sont détectées automatiquement par proximité:
- Distance euclidienne 3D < 3.0m
- Séparation temporelle > 30 poses
- Contrôlable via `--loop-sampling`

## Modèle de bruit

Le bruit gaussien (configurable via `--noise-sigma`) est appliqué aux mesures d'odométrie dans le repère local du robot:
- Bruit de position: 3D (x, y, z)
- Bruit d'orientation: 3D (roll, pitch, yaw) puis conversion en quaternion
- Les 5 premières poses sont sans bruit pour une ancre stable

## Fichiers générés

- `noise_3d.g2o`: Graphe de poses d'entrée avec odométrie bruitée et loop closures
- `opt_3d.g2o`: Sortie de g2o avec positions de vertex corrigées

## Métriques de performance

Le système calcule et affiche:

**Métriques de position:**
- RMSE (Root Mean Square Error) en mètres
- Erreur moyenne de position
- Erreur maximale de position

**Métriques d'orientation:**
- RMSE en radians (distance géodésique sur SO(3))
- Erreur moyenne d'orientation
- Erreur maximale d'orientation
- Versions en degrés pour lisibilité

**Pourcentage d'amélioration:** Montre la réduction de chaque métrique après optimisation

## Visualisation

Les graphiques 3D interactifs montrent:
- Trajectoires avec points et lignes
- Flèches d'orientation 3D (direction avant du repère local)
- Loop closures en cyan pointillé
- Comparaisons côte à côte des trajectoires
- Graphiques d'erreurs par pose (position et orientation)

## Notes de développement

### Système de coordonnées
Toutes les poses utilisent un repère monde 3D avec représentation `(x, y, z, qx, qy, qz, qw)` où le quaternion est au format `[x, y, z, w]` (convention scipy).

### Quaternions vs Euler
Les quaternions évitent le gimbal lock et permettent une interpolation stable. Les angles d'Euler sont utilisés uniquement pour:
- Génération de trajectoire (plus intuitif)
- Ajout de bruit (dans l'espace tangent)

### Matrices d'information
Nous utilisons des matrices d'information diagonales pour simplifier. Dans un système réel, ces matrices seraient calculées à partir des covariances des capteurs.

## Comparaison 2D vs 3D

| Aspect | 2D | 3D |
|--------|----|----|
| Dimensions de position | 2 (x, y) | 3 (x, y, z) |
| Dimensions d'orientation | 1 (θ) | 3 (quaternion/Euler) |
| Espace de configuration | SE(2) | SE(3) |
| Matrice d'information | 3×3 → 6 valeurs | 6×6 → 21 valeurs |
| Complexité calcul | O(n) | O(n) mais plus coûteux |
| Visualisation | 2D plots | 3D plots interactifs |

## Dépannage

**G2O ne trouve pas VERTEX_SE3:QUAT:**
- Vérifiez que g2o est compilé avec le support SE3
- Essayez `g2o --help` pour voir les types supportés

**Erreur de visualisation 3D:**
- Assurez-vous que matplotlib est installé avec support 3D
- Utilisez `--no-viz` pour désactiver la visualisation

**Optimisation lente:**
- Réduisez `--num-poses`
- Augmentez `--loop-sampling` pour moins de loop closures
- Utilisez `--no-viz` pour sauter la visualisation
