#!/bin/bash

# Script to download Khana dataset from Google Drive folder
# Folder link: https://drive.google.com/drive/folders/1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0
# Contains: dataset.tar.gz (6.43 GB), labels.txt, taxonomy.csv

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="/export/home/4prasad/piyush/Food_Vision_AI"
DATA_DIR="${PROJECT_ROOT}/data"
KHANA_DIR="${DATA_DIR}/khana_dataset"
DOWNLOAD_DIR="${DATA_DIR}/downloads"
FOLDER_ID="1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Downloading Khana Dataset from Google Drive${NC}"
echo -e "${GREEN}========================================${NC}"

# Create directories
mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${KHANA_DIR}"

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}gdown not found. Installing...${NC}"
    pip install gdown 2>/dev/null || \
    python3 -m pip install --user gdown 2>/dev/null || \
    python3 -m pip install --break-system-packages gdown 2>/dev/null || {
        echo -e "${RED}Failed to install gdown. Please install manually: pip install gdown${NC}"
        exit 1
    }
fi

echo -e "${GREEN}Downloading dataset from Google Drive folder...${NC}"
echo -e "${YELLOW}Folder ID: ${FOLDER_ID}${NC}"
echo -e "${YELLOW}This will download:${NC}"
echo -e "  - dataset.tar.gz (6.43 GB) - Main dataset"
echo -e "  - labels.txt - Class labels"
echo -e "  - taxonomy.csv - Taxonomy information"
echo -e "${YELLOW}This may take a while depending on your internet speed...${NC}"
echo ""

# Download the entire folder
echo -e "${GREEN}Downloading folder contents...${NC}"
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${DOWNLOAD_DIR}" || {
    echo -e "${RED}Error: Download failed.${NC}"
    echo -e "${YELLOW}Trying alternative method...${NC}"
    
    # Alternative: Try downloading folder as ZIP
    gdown --folder --remaining-ok "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${DOWNLOAD_DIR}/khana_folder" || {
        echo -e "${RED}Error: Both download methods failed.${NC}"
        echo -e "${YELLOW}Please try:${NC}"
        echo "  1. Make sure the folder is publicly accessible"
        echo "  2. Or download manually and extract to ${KHANA_DIR}"
        exit 1
    }
}

echo ""
echo -e "${GREEN}Download complete!${NC}"

# Check what was downloaded
echo -e "${GREEN}Checking downloaded files...${NC}"
ls -lh "${DOWNLOAD_DIR}"/* 2>/dev/null || ls -lh "${DOWNLOAD_DIR}"/khana_folder/* 2>/dev/null

# Find the dataset.tar.gz file
DATASET_FILE=$(find "${DOWNLOAD_DIR}" -name "dataset.tar.gz" -type f | head -1)

if [ -z "$DATASET_FILE" ]; then
    # Check in subfolder
    DATASET_FILE=$(find "${DOWNLOAD_DIR}" -name "dataset.tar.gz" -type f | head -1)
fi

if [ -n "$DATASET_FILE" ]; then
    echo ""
    echo -e "${GREEN}Found dataset file: ${DATASET_FILE}${NC}"
    echo -e "${GREEN}Extracting dataset...${NC}"
    
    # Extract to khana_dataset directory
    tar -xzf "${DATASET_FILE}" -C "${KHANA_DIR}" || {
        echo -e "${RED}Error: Failed to extract tar.gz file${NC}"
        exit 1
    }
    
    echo -e "${GREEN}Extraction complete!${NC}"
    
    # Move other files if they exist
    if [ -f "${DOWNLOAD_DIR}/labels.txt" ]; then
        cp "${DOWNLOAD_DIR}/labels.txt" "${KHANA_DIR}/"
        echo -e "${GREEN}Copied labels.txt${NC}"
    fi
    
    if [ -f "${DOWNLOAD_DIR}/taxonomy.csv" ]; then
        cp "${DOWNLOAD_DIR}/taxonomy.csv" "${KHANA_DIR}/"
        echo -e "${GREEN}Copied taxonomy.csv${NC}"
    fi
    
else
    echo -e "${YELLOW}Warning: dataset.tar.gz not found in downloaded files${NC}"
    echo -e "${YELLOW}Please check the download directory: ${DOWNLOAD_DIR}${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dataset location: ${KHANA_DIR}${NC}"
echo ""
echo -e "${YELLOW}Next step: Run the setup script${NC}"
echo "  python3 scripts/setup_khana_dataset.py"

