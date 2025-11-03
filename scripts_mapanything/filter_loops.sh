#!/bin/bash
# Script wrapper pour filtrer les loops validées
# Usage: ./filter_loops.sh [OPTIONS]

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Fonction d'aide
show_help() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}Script de Filtrage des Loops Validées${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Ce script filtre les loops qui ont passé le test de confiance"
    echo "et copie les images correspondantes dans un nouveau dossier."
    echo ""
    echo "Options via variables d'environnement:"
    echo "  CSV_FILE        - Fichier CSV source (défaut: loop_poses.csv)"
    echo "  IMAGE_FOLDER    - Dossier contenant les images (défaut: /home/ivm/loc/loop_pairs/good_loops)"
    echo "  OUTPUT_FOLDER   - Dossier de sortie (défaut: validated_loops)"
    echo "  MIN_CONFIDENCE  - Seuil minimum de confiance (défaut: 0.0)"
    echo "  LIST_FILE       - Fichier texte de sortie (défaut: validated_loops.txt)"
    echo ""
    echo "Exemples:"
    echo "  # Filtrer avec seuil 0.6"
    echo "  MIN_CONFIDENCE=0.6 $0"
    echo ""
    echo "  # Utiliser un autre CSV et dossier de sortie"
    echo "  CSV_FILE=results.csv OUTPUT_FOLDER=good_loops $0"
    echo ""
    echo "  # Filtrer strictement (0.8) et sauver dans custom_folder"
    echo "  MIN_CONFIDENCE=0.8 OUTPUT_FOLDER=strict_loops $0"
    echo ""
    echo "  # Tout garder (pas de filtrage)"
    echo "  MIN_CONFIDENCE=0.0 $0"
    echo ""
}

# Vérifier l'aide
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Configuration par défaut
CSV_FILE="${CSV_FILE:-loop_poses.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/ivm/loc/loop_pairs/good_loops}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-validated_loops}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.0}"
LIST_FILE="${LIST_FILE:-validated_loops.txt}"

# Afficher la configuration
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}Configuration du filtrage${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "CSV source:        ${BLUE}$CSV_FILE${NC}"
echo -e "Images source:     ${BLUE}$IMAGE_FOLDER${NC}"
echo -e "Dossier sortie:    ${BLUE}$OUTPUT_FOLDER${NC}"
echo -e "Seuil confiance:   ${BLUE}$MIN_CONFIDENCE${NC}"
echo -e "Fichier liste:     ${BLUE}$LIST_FILE${NC}"
echo ""

# Vérifier que les fichiers existent
if [[ ! -f "$CSV_FILE" ]]; then
    echo -e "${RED}❌ Erreur: Le fichier CSV '$CSV_FILE' n'existe pas${NC}"
    echo ""
    echo "Solutions:"
    echo "  1. Vérifiez que l'estimation a été lancée: ./run_loop_estimation.sh"
    echo "  2. Ou spécifiez le bon fichier: CSV_FILE=other.csv $0"
    echo ""
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo -e "${RED}❌ Erreur: Le dossier '$IMAGE_FOLDER' n'existe pas${NC}"
    exit 1
fi

# Construire la commande
PYTHON_CMD="python scripts/filter_validated_loops.py"
PYTHON_CMD="$PYTHON_CMD --csv \"$CSV_FILE\""
PYTHON_CMD="$PYTHON_CMD --image_folder \"$IMAGE_FOLDER\""
PYTHON_CMD="$PYTHON_CMD --output_folder \"$OUTPUT_FOLDER\""
PYTHON_CMD="$PYTHON_CMD --min_confidence $MIN_CONFIDENCE"
PYTHON_CMD="$PYTHON_CMD --list_file \"$LIST_FILE\""

# Afficher la commande
echo -e "${BLUE}Commande:${NC}"
echo "$PYTHON_CMD"
echo ""

# Demander confirmation
read -p "Lancer le filtrage ? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️  Annulé par l'utilisateur${NC}"
    exit 0
fi

echo -e "${GREEN}🚀 Lancement du filtrage...${NC}"
echo ""

# Exécuter
eval $PYTHON_CMD

# Vérifier le résultat
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}✅ Filtrage terminé avec succès !${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""
    echo "Résultats:"
    echo -e "  📁 Images validées: ${BLUE}$OUTPUT_FOLDER/${NC}"
    echo -e "  📝 Liste texte:     ${BLUE}$LIST_FILE${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}==================================================================${NC}"
    echo -e "${RED}❌ Erreur lors du filtrage${NC}"
    echo -e "${RED}==================================================================${NC}"
    exit 1
fi
