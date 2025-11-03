#!/bin/bash
# Script pour lancer l'estimation de poses relatives entre paires query/loop
# Usage: ./run_loop_estimation.sh [MODE]
# Modes disponibles: basic, fast, accurate, memory-save, custom

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}Script d'Estimation de Poses Relatives pour Loop Closure${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes disponibles:"
    echo "  basic       - Mode basique sans filtrage (défaut)"
    echo "  fast        - Mode rapide avec confiance faible (seuil 0.3)"
    echo "  accurate    - Mode précis avec confiance élevée (seuil 0.6)"
    echo "  memory-save - Mode économe en mémoire (pour grandes images)"
    echo "  apache      - Utilise le modèle Apache au lieu de CC-BY-NC"
    echo "  cpu         - Force l'utilisation du CPU (très lent, sans GPU)"
    echo "  custom      - Mode personnalisé (modifiez les variables dans le script)"
    echo ""
    echo "Options requises (définies dans le script ou via variables d'environnement):"
    echo "  IMAGE_FOLDER - Dossier contenant les paires query/loop"
    echo "  OUTPUT_CSV   - Fichier CSV de sortie (défaut: loop_poses.csv)"
    echo ""
    echo "Variables d'environnement optionnelles:"
    echo "  CONF_THRESHOLD - Seuil de confiance personnalisé (0.0 - 1.0)"
    echo "  USE_CPU        - Forcer CPU même en mode GPU (export USE_CPU=1)"
    echo "  BATCH_SIZE     - Nombre de paires par batch (default: 1)"
    echo "  MAX_PAIRS      - Limiter le nombre de paires à traiter (pour tests)"
    echo "  FILTER_LOOPS   - Filtrer et copier les loops validées (export FILTER_LOOPS=1)"
    echo "  OUTPUT_FOLDER  - Dossier pour images validées (défaut: validated_loops)"
    echo "  MIN_CONFIDENCE - Seuil pour filtrage (défaut: utilise CONF_THRESHOLD)"
    echo ""
    echo "Exemples:"
    echo "  $0 basic"
    echo "  $0 accurate"
    echo "  IMAGE_FOLDER=/path/to/images $0 fast"
    echo "  CONF_THRESHOLD=0.7 $0 custom"
    echo "  USE_CPU=1 $0 apache              # Apache sur CPU"
    echo "  BATCH_SIZE=4 $0 accurate         # Batch de 4 paires"
    echo "  MAX_PAIRS=10 $0 basic            # Tester sur 10 paires seulement"
    echo "  $0 cpu                           # Mode CPU explicite"
    echo "  FILTER_LOOPS=1 $0 accurate       # Estimation + filtrage automatique"
    echo "  FILTER_LOOPS=1 MIN_CONFIDENCE=0.7 $0 custom  # Filtrage avec seuil 0.7"
    echo ""
}

# Vérifier si l'aide est demandée
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# =============================================================================
# CONFIGURATION - MODIFIEZ CES VARIABLES SELON VOS BESOINS
# =============================================================================

# Dossier contenant les images (query_*.jpg et loop_*.jpg)
# Vous pouvez aussi définir cette variable dans votre terminal: export IMAGE_FOLDER=/path/to/images
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/ivm/loc/loop_pairs/bad_loops}"

# Fichier CSV de sortie
OUTPUT_CSV="${OUTPUT_CSV:-loop_poses.csv}"

# Modèle à utiliser (ne changez que si vous avez un modèle local)
MODEL="${MODEL:-facebook/map-anything}"

# Filtrage automatique des loops validées
FILTER_LOOPS="${FILTER_LOOPS:-0}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-validated_loops}"
LIST_FILE="${LIST_FILE:-validated_loops.txt}"

# =============================================================================
# NE MODIFIEZ PAS EN DESSOUS SAUF SI VOUS SAVEZ CE QUE VOUS FAITES
# =============================================================================

# Déterminer le mode
MODE="${1:-basic}"

# Vérifier que IMAGE_FOLDER est défini et existe
if [[ "$IMAGE_FOLDER" == "/path/to/your/images" ]]; then
    echo -e "${RED}❌ Erreur: Vous devez définir IMAGE_FOLDER${NC}"
    echo -e "${YELLOW}Solutions:${NC}"
    echo "  1. Modifiez la variable IMAGE_FOLDER dans ce script"
    echo "  2. Ou utilisez: IMAGE_FOLDER=/path/to/images $0 $MODE"
    echo ""
    exit 1
fi

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo -e "${RED}❌ Erreur: Le dossier $IMAGE_FOLDER n'existe pas${NC}"
    exit 1
fi

# Afficher la configuration
echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}Configuration de l'estimation de poses${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "Mode:              ${BLUE}$MODE${NC}"
echo -e "Dossier d'images:  ${BLUE}$IMAGE_FOLDER${NC}"
echo -e "Fichier de sortie: ${BLUE}$OUTPUT_CSV${NC}"
echo -e "Modèle:            ${BLUE}$MODEL${NC}"
if [[ "$FILTER_LOOPS" == "1" ]]; then
    echo -e "Filtrage:          ${GREEN}ACTIVÉ${NC}"
    echo -e "Dossier validées:  ${BLUE}$OUTPUT_FOLDER${NC}"
else
    echo -e "Filtrage:          ${YELLOW}DÉSACTIVÉ${NC} (utilisez FILTER_LOOPS=1 pour activer)"
fi
echo ""

# Construire la commande selon le mode
PYTHON_CMD="python scripts/estimate_loop_poses.py --image_folder \"$IMAGE_FOLDER\" --output \"$OUTPUT_CSV\""

# Vérifier si on doit forcer le CPU
if [[ "$USE_CPU" == "1" || "$MODE" == "cpu" ]]; then
    PYTHON_CMD="$PYTHON_CMD --cpu"
fi

# Ajouter le batch_size si défini
if [[ -n "$BATCH_SIZE" ]]; then
    PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
fi

# Ajouter max_pairs si défini
if [[ -n "$MAX_PAIRS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max_pairs $MAX_PAIRS"
fi

# Déterminer le seuil de confiance
# Si CONF_THRESHOLD est défini par l'utilisateur, l'utiliser prioritairement
if [[ -n "$CONF_THRESHOLD" ]]; then
    THRESHOLD="$CONF_THRESHOLD"
    THRESHOLD_SOURCE="utilisateur"
else
    # Sinon utiliser le seuil par défaut du mode
    case "$MODE" in
        fast) THRESHOLD="0.3" ;;
        accurate) THRESHOLD="0.6" ;;
        memory-save|apache|cpu) THRESHOLD="0.5" ;;
        custom) THRESHOLD="0.5" ;;
        basic) THRESHOLD="" ;;  # Pas de seuil en mode basic
    esac
    THRESHOLD_SOURCE="mode $MODE"
fi

case "$MODE" in
    basic)
        echo -e "${YELLOW}Mode BASIC: Estimation simple sans filtrage${NC}"
        if [[ -n "$THRESHOLD" ]]; then
            echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        else
            echo "  - Pas de seuil de confiance"
        fi
        echo "  - Inférence standard (rapide)"
        echo "  - Modèle CC-BY-NC 4.0"
        echo ""
        ;;

    fast)
        echo -e "${YELLOW}Mode FAST: Rapide avec filtrage léger${NC}"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo "  - Inférence standard (rapide)"
        echo "  - Modèle CC-BY-NC 4.0"
        echo ""
        ;;

    accurate)
        echo -e "${YELLOW}Mode ACCURATE: Précis avec filtrage strict${NC}"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo "  - Inférence standard"
        echo "  - Modèle CC-BY-NC 4.0"
        echo ""
        ;;

    memory-save)
        echo -e "${YELLOW}Mode MEMORY-SAVE: Économe en mémoire${NC}"
        echo "  - Inférence économe en mémoire (plus lent mais utilise moins de VRAM)"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo "  - Recommandé pour: grandes images, GPU avec peu de VRAM"
        echo ""
        PYTHON_CMD="$PYTHON_CMD --memory_efficient"
        ;;

    apache)
        echo -e "${YELLOW}Mode APACHE: Modèle Apache 2.0${NC}"
        echo "  - Utilise facebook/map-anything-apache"
        echo "  - Licence Apache 2.0 (usage commercial autorisé)"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo ""
        PYTHON_CMD="$PYTHON_CMD --apache"
        ;;

    cpu)
        echo -e "${YELLOW}Mode CPU: Force l'utilisation du CPU${NC}"
        echo "  - ⚠️  TRÈS LENT: 10-50x plus lent que GPU"
        echo "  - Utilise uniquement le CPU (pas de CUDA requis)"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo "  - Utile si: pas de GPU, problèmes CUDA, debugging"
        echo ""
        # --cpu déjà ajouté plus haut
        ;;

    custom)
        echo -e "${YELLOW}Mode CUSTOM: Configuration personnalisée${NC}"
        echo "  - Seuil de confiance: $THRESHOLD (défini par $THRESHOLD_SOURCE)"
        echo "  - Vous pouvez modifier ce mode dans le script"
        echo ""

        # Ajoutez vos options personnalisées ici
        # Exemples:
        # PYTHON_CMD="$PYTHON_CMD --memory_efficient"
        # PYTHON_CMD="$PYTHON_CMD --apache"
        ;;

    *)
        echo -e "${RED}❌ Erreur: Mode '$MODE' inconnu${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

# Ajouter le seuil de confiance à la commande si défini
if [[ -n "$THRESHOLD" ]]; then
    PYTHON_CMD="$PYTHON_CMD --confidence_threshold $THRESHOLD"
fi

# Afficher la commande qui va être exécutée
echo -e "${BLUE}Commande:${NC}"
echo "$PYTHON_CMD"
echo ""

# Demander confirmation
read -p "Lancer l'estimation ? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️  Annulé par l'utilisateur${NC}"
    exit 0
fi

echo -e "${GREEN}🚀 Lancement de l'estimation...${NC}"
echo ""

# Exécuter la commande
eval $PYTHON_CMD

# Vérifier le code de retour
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}✅ Estimation terminée avec succès !${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "Résultats sauvegardés dans: ${BLUE}$OUTPUT_CSV${NC}"
    echo ""

    # Filtrage automatique si activé
    if [[ "$FILTER_LOOPS" == "1" ]]; then
        echo -e "${BLUE}==================================================================${NC}"
        echo -e "${BLUE}🔍 Filtrage des loops validées${NC}"
        echo -e "${BLUE}==================================================================${NC}"

        # Déterminer le seuil de confiance pour le filtrage
        # Si MIN_CONFIDENCE est défini, l'utiliser, sinon utiliser le seuil d'estimation
        if [[ -n "$MIN_CONFIDENCE" ]]; then
            FILTER_THRESHOLD="$MIN_CONFIDENCE"
            FILTER_SOURCE="MIN_CONFIDENCE"
        elif [[ -n "$THRESHOLD" ]]; then
            FILTER_THRESHOLD="$THRESHOLD"
            FILTER_SOURCE="seuil d'estimation"
        else
            FILTER_THRESHOLD="0.0"
            FILTER_SOURCE="défaut (pas de filtrage)"
        fi

        echo -e "Seuil de filtrage: ${BLUE}$FILTER_THRESHOLD${NC} (source: $FILTER_SOURCE)"
        echo ""

        # Lancer le filtrage
        python scripts/filter_validated_loops.py \
            --csv "$OUTPUT_CSV" \
            --image_folder "$IMAGE_FOLDER" \
            --output_folder "$OUTPUT_FOLDER" \
            --min_confidence "$FILTER_THRESHOLD" \
            --list_file "$LIST_FILE"

        if [ $? -eq 0 ]; then
            echo ""
            echo -e "${GREEN}✅ Filtrage terminé${NC}"
            echo -e "Images validées: ${BLUE}$OUTPUT_FOLDER/${NC}"
            echo -e "Liste texte:     ${BLUE}$LIST_FILE${NC}"
            echo ""
        else
            echo ""
            echo -e "${YELLOW}⚠️  Erreur lors du filtrage (estimation OK)${NC}"
            echo ""
        fi
    fi

    echo "Pour analyser les résultats:"
    echo "  - Ouvrir le CSV dans Excel/LibreOffice"
    echo "  - Ou utiliser Python/pandas pour une analyse plus poussée"
    if [[ "$FILTER_LOOPS" == "1" ]]; then
        echo "  - Images validées disponibles dans: $OUTPUT_FOLDER/"
    fi
    echo ""
else
    echo ""
    echo -e "${RED}==================================================================${NC}"
    echo -e "${RED}❌ Erreur lors de l'estimation${NC}"
    echo -e "${RED}==================================================================${NC}"
    exit 1
fi
