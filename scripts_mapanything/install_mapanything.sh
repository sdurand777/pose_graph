#!/bin/bash
# Script d'installation de MapAnything pour pose_graph
# Supporte installation venv (recommandé) ou system-wide

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chemin vers le repository MapAnything
MAPANYTHING_REPO="/home/ivm/map-anything"

# Nom du virtualenv (si venv est choisi)
VENV_NAME="mapanything_env"

# Version Python minimum requise
REQUIRED_PYTHON_VERSION="3.10"

# =============================================================================
# FONCTIONS
# =============================================================================

show_header() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}        Installation de MapAnything pour pose_graph${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
}

show_help() {
    show_header
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Modes disponibles:"
    echo "  venv        - Installation dans un virtualenv Python (RECOMMANDÉ)"
    echo "  system      - Installation system-wide avec pip (nécessite sudo)"
    echo "  --help, -h  - Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  $0 venv      # Installation dans virtualenv (défaut)"
    echo "  $0 system    # Installation system-wide"
    echo ""
}

check_python_version() {
    echo -e "${CYAN}🔍 Vérification de la version Python...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 n'est pas installé${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)

    echo -e "   Version détectée: ${GREEN}$PYTHON_VERSION${NC}"

    # Comparer les versions
    if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_PYTHON_VERSION" | sort -V | head -1) != "$REQUIRED_PYTHON_VERSION" ]]; then
        echo -e "${RED}❌ Python >= $REQUIRED_PYTHON_VERSION requis (trouvé: $PYTHON_VERSION)${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Version Python compatible${NC}"
    echo ""
}

check_cuda() {
    echo -e "${CYAN}🔍 Vérification de CUDA...${NC}"

    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo -e "   CUDA détecté: ${GREEN}$CUDA_VERSION${NC}"
        echo -e "${GREEN}✅ CUDA disponible - Installation de PyTorch avec support GPU${NC}"
        CUDA_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  CUDA non détecté - Installation de PyTorch CPU-only${NC}"
        CUDA_AVAILABLE=false
    fi
    echo ""
}

check_mapanything_repo() {
    echo -e "${CYAN}🔍 Vérification du repository MapAnything...${NC}"

    if [[ ! -d "$MAPANYTHING_REPO" ]]; then
        echo -e "${RED}❌ Repository MapAnything non trouvé: $MAPANYTHING_REPO${NC}"
        echo ""
        echo "Pour installer MapAnything, clonez d'abord le repository:"
        echo "  cd /home/ivm"
        echo "  git clone https://github.com/facebookresearch/map-anything.git"
        exit 1
    fi

    if [[ ! -f "$MAPANYTHING_REPO/pyproject.toml" ]]; then
        echo -e "${RED}❌ pyproject.toml non trouvé dans $MAPANYTHING_REPO${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Repository MapAnything trouvé: $MAPANYTHING_REPO${NC}"
    echo ""
}

install_venv() {
    echo -e "${CYAN}📦 Installation dans virtualenv...${NC}"
    echo ""

    # Créer le virtualenv s'il n'existe pas
    if [[ -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}⚠️  Virtualenv '$VENV_NAME' existe déjà${NC}"
        read -p "Supprimer et recréer? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}🗑️  Suppression de l'ancien virtualenv...${NC}"
            rm -rf "$VENV_NAME"
        else
            echo -e "${YELLOW}⚠️  Utilisation du virtualenv existant${NC}"
        fi
    fi

    if [[ ! -d "$VENV_NAME" ]]; then
        echo -e "${CYAN}🔨 Création du virtualenv '$VENV_NAME'...${NC}"
        python3 -m venv "$VENV_NAME"
        echo -e "${GREEN}✅ Virtualenv créé${NC}"
    fi

    echo ""
    echo -e "${CYAN}🔨 Activation du virtualenv...${NC}"
    source "$VENV_NAME/bin/activate"
    echo -e "${GREEN}✅ Virtualenv activé${NC}"
    echo ""

    # Mettre à jour pip
    echo -e "${CYAN}🔨 Mise à jour de pip...${NC}"
    pip install --upgrade pip setuptools wheel
    echo ""

    # Installer PyTorch
    install_pytorch

    # Installer MapAnything
    install_mapanything

    echo ""
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}✅ Installation terminée avec succès!${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Pour utiliser MapAnything:${NC}"
    echo -e "  1. Activez le virtualenv:"
    echo -e "     ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo ""
    echo -e "  2. Lancez le script d'estimation:"
    echo -e "     ${CYAN}./run_loop_estimation.sh accurate${NC}"
    echo ""
    echo -e "  3. Pour désactiver le virtualenv:"
    echo -e "     ${CYAN}deactivate${NC}"
    echo ""

    # Créer un fichier d'activation rapide
    cat > activate_mapanything.sh <<'EOF'
#!/bin/bash
# Script d'activation rapide du virtualenv MapAnything
source mapanything_env/bin/activate
echo "✅ Virtualenv MapAnything activé"
echo "Vous pouvez maintenant utiliser: ./run_loop_estimation.sh"
EOF
    chmod +x activate_mapanything.sh

    echo -e "${CYAN}💡 Astuce: Utilisez ${GREEN}./activate_mapanything.sh${CYAN} pour activer rapidement l'environnement${NC}"
    echo ""
}

install_system() {
    echo -e "${CYAN}📦 Installation system-wide...${NC}"
    echo ""

    echo -e "${YELLOW}⚠️  Attention: Installation system-wide${NC}"
    echo -e "${YELLOW}   Cela modifiera vos paquets Python globaux${NC}"
    echo -e "${YELLOW}   Il est recommandé d'utiliser un virtualenv à la place${NC}"
    echo ""
    read -p "Continuer avec l'installation system-wide? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}❌ Installation annulée${NC}"
        exit 0
    fi

    echo ""
    echo -e "${CYAN}🔨 Mise à jour de pip...${NC}"
    python3 -m pip install --upgrade pip setuptools wheel
    echo ""

    # Installer PyTorch
    install_pytorch

    # Installer MapAnything
    install_mapanything

    echo ""
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}✅ Installation system-wide terminée avec succès!${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Pour utiliser MapAnything:${NC}"
    echo -e "  ${CYAN}./run_loop_estimation.sh accurate${NC}"
    echo ""
}

install_pytorch() {
    echo -e "${CYAN}🔨 Installation de PyTorch...${NC}"

    if [[ "$CUDA_AVAILABLE" == true ]]; then
        echo -e "   Avec support CUDA..."
        # PyTorch avec CUDA 12.1 (compatible avec CUDA 12.6)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo -e "   CPU-only..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    echo -e "${GREEN}✅ PyTorch installé${NC}"
    echo ""
}

install_mapanything() {
    echo -e "${CYAN}🔨 Installation de MapAnything et ses dépendances...${NC}"
    echo ""

    # Installer les dépendances de base
    echo -e "${CYAN}   📦 Installation des dépendances de base...${NC}"
    pip install -e "$MAPANYTHING_REPO"

    # Installer gradio pour les scripts de demo potentiels
    echo -e "${CYAN}   📦 Installation de gradio (optionnel)...${NC}"
    pip install -e "$MAPANYTHING_REPO[gradio]"

    # Installer pandas pour l'analyse de données
    echo -e "${CYAN}   📦 Installation de pandas (optionnel)...${NC}"
    pip install -e "$MAPANYTHING_REPO[data]"

    echo -e "${GREEN}✅ MapAnything installé depuis: $MAPANYTHING_REPO${NC}"
    echo ""
}

verify_installation() {
    echo -e "${CYAN}🔍 Vérification de l'installation...${NC}"
    echo ""

    # Vérifier l'import de mapanything
    if python3 -c "import mapanything" 2>/dev/null; then
        echo -e "${GREEN}✅ Module mapanything importable${NC}"
    else
        echo -e "${RED}❌ Erreur: impossible d'importer mapanything${NC}"
        return 1
    fi

    # Vérifier PyTorch
    if python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        echo -e "${GREEN}✅ PyTorch installé${NC}"

        # Vérifier CUDA
        CUDA_STATUS=$(python3 -c "import torch; print('available' if torch.cuda.is_available() else 'unavailable')" 2>/dev/null)
        if [[ "$CUDA_STATUS" == "available" ]]; then
            echo -e "${GREEN}✅ CUDA disponible dans PyTorch${NC}"
        else
            echo -e "${YELLOW}⚠️  CUDA non disponible dans PyTorch (mode CPU)${NC}"
        fi
    else
        echo -e "${RED}❌ Erreur: impossible d'importer torch${NC}"
        return 1
    fi

    # Vérifier les autres dépendances importantes
    for pkg in "huggingface_hub" "hydra" "tqdm" "opencv-python-headless"; do
        PKG_NAME=$(echo "$pkg" | sed 's/-/_/g')
        if python3 -c "import $PKG_NAME" 2>/dev/null; then
            echo -e "${GREEN}✅ $pkg installé${NC}"
        else
            echo -e "${YELLOW}⚠️  $pkg non installé${NC}"
        fi
    done

    echo ""
    echo -e "${GREEN}✅ Vérification terminée${NC}"
    echo ""
}

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

# Vérifier l'aide
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Déterminer le mode
MODE="${1:-venv}"

show_header

# Vérifications préliminaires
check_python_version
check_cuda
check_mapanything_repo

# Installation selon le mode
case "$MODE" in
    venv)
        install_venv
        ;;
    system)
        install_system
        ;;
    *)
        echo -e "${RED}❌ Mode inconnu: $MODE${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

# Vérifier l'installation
verify_installation

# Message final
echo -e "${CYAN}💡 Conseil: Consultez $MAPANYTHING_REPO/README.md pour plus d'informations${NC}"
echo -e "${CYAN}💡 Scripts disponibles dans $MAPANYTHING_REPO/scripts/${NC}"
echo ""
