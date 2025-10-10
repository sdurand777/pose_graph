# Comparaison des implémentations 2D et 3D

Ce document compare les deux implémentations du système de pose graph SLAM.

## Vue d'ensemble

| Aspect | 2D (SE2) | 3D (SE3) |
|--------|----------|----------|
| **Fichier principal** | `main.py` | `main_3d.py` |
| **Espace de poses** | SE(2) | SE(3) |
| **Dimensions** | 3 (x, y, θ) | 7 (x, y, z, qx, qy, qz, qw) |
| **Format G2O** | VERTEX_SE2, EDGE_SE2 | VERTEX_SE3:QUAT, EDGE_SE3:QUAT |

## Modules et correspondances

### 1. Génération de trajectoires

| 2D | 3D | Différences clés |
|----|----|--------------------|
| `trajectory.py` | `trajectory_3d.py` | - Ajout de coordonnée Z<br>- Quaternions au lieu d'angles<br>- Ajout de roll/pitch/yaw |
| `genTraj()` | `genTraj3D()` | - Trajectoire avec variations d'altitude<br>- Banking dans les virages |
| `genTrajSquareSpiral()` | `genTrajSpiralHelix3D()` | - Spirale hélicoïdale ascendante |
| `addNoise()` | `addNoise3D()` | - Bruit 3D sur position et rotation |

**Exemple de pose:**
- **2D**: `(x=-5.0, y=-8.0, θ=0.0)`
- **3D**: `(x=-5.0, y=-8.0, z=0.0, qx=0.0, qy=0.05, qz=0.0, qw=0.999)`

### 2. I/O des fichiers G2O

| 2D | 3D | Différences clés |
|----|----|--------------------|
| `g2o_io.py` | `g2o_io_3d.py` | - Format VERTEX_SE3:QUAT<br>- Matrices d'information 6×6 |
| `writeOdom()` | `writeOdom3D()` | - 21 valeurs d'information au lieu de 6 |
| `writeLoop()` | `writeLoop3D()` | - Distance euclidienne 3D |
| `readG2o()` | `readG2o3D()` | - Parse les quaternions |

**Format des vertices:**
```
2D: VERTEX_SE2 0 -5.0 -8.0 0.0
3D: VERTEX_SE3:QUAT 0 -5.0 -8.0 0.0 0.0 0.05 0.0 0.999
```

**Format des edges:**
```
2D: EDGE_SE2 0 1 0.45 0.0 0.0 500 0 0 500 0 500
3D: EDGE_SE3:QUAT 0 1 0.45 0.0 0.1 0.0 0.0 0.0 1.0 500 0 0 0 0 0 500 0 0 0 0 500 0 0 0 500 0 0 500 0 500
```

### 3. Visualisation

| 2D | 3D | Différences clés |
|----|----|--------------------|
| `visualization.py` | `visualization_3d.py` | - Plots 3D interactifs<br>- Flèches d'orientation 3D |
| `draw()` | `draw3D()` | - `ax = fig.add_subplot(111, projection='3d')` |
| `drawTwo()` | `drawTwo3D()` | - Ajout de l'axe Z |
| `drawThree()` | `drawThree3D()` | - Visualisation 3D complète |
| `plotErrors()` | `plotErrors3D()` | - Même structure |

**Différence majeure:**
```python
# 2D
ax.plot(X, Y, 'ro')

# 3D
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, 'ro')
```

### 4. Métriques

| 2D | 3D | Différences clés |
|----|----|--------------------|
| `metrics.py` | `metrics_3d.py` | - Distance euclidienne 3D<br>- Distance géodésique sur SO(3) |
| `compute_pose_errors()` | `compute_pose_errors_3d()` | - Erreur d'orientation via quaternions |

**Calcul d'erreur d'orientation:**
```python
# 2D: Simple différence d'angles
diff = THETA_test[i] - THETA_ref[i]
error = np.arctan2(np.sin(diff), np.cos(diff))

# 3D: Distance géodésique
r_ref = Rotation.from_quat([QX_ref[i], QY_ref[i], QZ_ref[i], QW_ref[i]])
r_test = Rotation.from_quat([QX_test[i], QY_test[i], QZ_test[i], QW_test[i]])
r_diff = r_ref.inv() * r_test
error = r_diff.magnitude()  # Angle de rotation
```

### 5. Optimisation

| 2D | 3D | Différences clés |
|----|----|--------------------|
| `pose_graph.py` | `pose_graph_3d.py` | - Même commande g2o |
| `optimize()` | `optimize3D()` | - Fichiers différents |

```python
# 2D
cmd = "g2o -o opt.g2o noise.g2o"

# 3D
cmd = "g2o -o opt_3d.g2o noise_3d.g2o"
```

## Transformations mathématiques

### Matrices de transformation

**2D (SE2) - Matrice 3×3:**
```
T = [cos(θ)  -sin(θ)   x]
    [sin(θ)   cos(θ)   y]
    [0        0        1]
```

**3D (SE3) - Matrice 4×4:**
```
T = [R₀₀  R₀₁  R₀₂  x]
    [R₁₀  R₁₁  R₁₂  y]
    [R₂₀  R₂₁  R₂₂  z]
    [0    0    0    1]
```
où R est la matrice de rotation 3×3 calculée depuis le quaternion.

### Composition de transformations

**2D:**
```python
T1_w = [[cos(θ₁), -sin(θ₁), x₁],
        [sin(θ₁),  cos(θ₁), y₁],
        [0,        0,       1]]
```

**3D:**
```python
r1 = Rotation.from_quat([qx, qy, qz, qw])
T1_w = np.eye(4)
T1_w[:3, :3] = r1.as_matrix()
T1_w[:3, 3] = [x, y, z]
```

## Détection de loop closures

| Aspect | 2D | 3D |
|--------|----|----|
| **Distance** | Euclidienne 2D | Euclidienne 3D |
| **Seuil** | < 2.5m | < 3.0m |
| **Séparation temporelle** | > 30 poses | > 30 poses |

```python
# 2D
dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)

# 3D
dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2 + (Z[i] - Z[j])**2)
```

## Ajout de bruit

### 2D
```python
# Bruit dans le repère local
xNoise = np.random.normal(0, noise_sigma)
yNoise = np.random.normal(0, noise_sigma)
tNoise = np.random.normal(0, noise_sigma)
```

### 3D
```python
# Bruit de position 3D
pos_noise = np.random.normal(0, noise_sigma, 3)

# Bruit de rotation (angles d'Euler)
rot_noise = np.random.normal(0, noise_sigma, 3)

# Application via composition de rotations
del_rotN = del_rot * Rotation.from_euler('xyz', rot_noise)
```

## Complexité computationnelle

| Opération | 2D | 3D |
|-----------|----|----|
| **Taille de pose** | 3 | 7 |
| **Multiplication de matrice** | 3×3 | 4×4 |
| **Normalisation quaternion** | N/A | Oui |
| **Variables d'optimisation** | 3n | 6n (translation + rotation) |

## Dépendances additionnelles

### 2D
```python
import numpy as np
import math
import matplotlib.pyplot as plt
```

### 3D
```python
import numpy as np
from scipy.spatial.transform import Rotation  # Pour quaternions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Pour plots 3D
```

## Fichiers générés

| 2D | 3D |
|----|-----|
| `noise.g2o` | `noise_3d.g2o` |
| `opt.g2o` | `opt_3d.g2o` |

## Exemples d'utilisation

### Exécution basique
```bash
# 2D
python main.py --seed 42

# 3D
python main_3d.py --seed 42
```

### Trajectoire spirale
```bash
# 2D - Square spiral
python main.py --trajectory-type square-spiral --seed 42

# 3D - Helical spiral
python main_3d.py --trajectory-type helix --seed 42
```

### Sans visualisation
```bash
# 2D
python main.py --seed 42 --no-viz

# 3D
python main_3d.py --seed 42 --no-viz
```

## Métriques de sortie

Les deux versions affichent:
- RMSE position/orientation
- Mean error
- Max error
- Pourcentage d'amélioration

**Différence**: Les erreurs d'orientation 3D sont également affichées en degrés pour meilleure lisibilité.

## Quand utiliser quelle version ?

### Utiliser la version 2D si:
- Robot à roues sur surface plane
- Drone avec altitude fixe
- Mouvement planaire seulement
- Prototypage rapide
- Calculs plus rapides nécessaires

### Utiliser la version 3D si:
- Drone avec variation d'altitude
- Robot sous-marin
- Mouvement 3D complet nécessaire
- Besoin de pitch/roll/yaw
- Application réaliste pour la plupart des robots

## Performance

Sur un système moderne:
- **2D**: ~0.5-1 seconde pour optimisation
- **3D**: ~1-3 secondes pour optimisation

La différence vient de:
1. Plus de variables (6 au lieu de 3 par pose)
2. Matrices d'information plus grandes (6×6 vs 3×3)
3. Calculs de quaternions plus coûteux

## Conversion 2D → 3D

Pour convertir un système 2D existant en 3D:

1. **Poses**: Ajouter Z et convertir θ en quaternion
```python
# 2D → 3D
z = 0.0  # ou hauteur appropriée
qx, qy, qz, qw = quaternion_from_euler(0, 0, theta)
```

2. **Transformations**: Utiliser matrices 4×4 au lieu de 3×3

3. **Visualisation**: Utiliser `projection='3d'`

4. **Format G2O**: Remplacer SE2 par SE3:QUAT

5. **Erreurs**: Ajouter Z dans distance euclidienne, utiliser distance géodésique pour orientation
