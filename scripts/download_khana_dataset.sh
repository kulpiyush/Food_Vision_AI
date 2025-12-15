#!/bin/bash

# Script to download and set up the Khana dataset for Food Vision AI
# This will REPLACE the current dataset completely
# Usage: ./download_khana_dataset.sh <GOOGLE_DRIVE_FILE_ID>
# Or: ./download_khana_dataset.sh "https://drive.google.com/file/d/<FILE_ID>/view"

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the file ID from command line argument
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <GOOGLE_DRIVE_FILE_ID> or <GOOGLE_DRIVE_URL>${NC}"
    echo "Example: $0 1ABC123xyz..."
    echo "Or: $0 'https://drive.google.com/file/d/1ABC123xyz.../view'"
    exit 1
fi

FILE_ID_OR_URL="$1"
PROJECT_ROOT="/export/home/4prasad/piyush/Food_Vision_AI"
DATA_DIR="${PROJECT_ROOT}/data"
KHANA_DIR="${DATA_DIR}/khana_dataset"
TRAINING_DATA_DIR="${DATA_DIR}/training_data"
DOWNLOAD_DIR="${DATA_DIR}/downloads"
ZIP_FILE="${DOWNLOAD_DIR}/khana.zip"
BACKUP_DIR="${DATA_DIR}/old_dataset_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Khana Dataset Setup - Replacing Old Dataset${NC}"
echo -e "${GREEN}========================================${NC}"

    # Step 1: Backup old dataset (if exists)
    if [ -d "${TRAINING_DATA_DIR}" ] && [ "$(ls -A ${TRAINING_DATA_DIR} 2>/dev/null)" ]; then
        echo -e "${YELLOW}Backing up old dataset to: ${BACKUP_DIR}${NC}"
        mkdir -p "${BACKUP_DIR}"
        cp -r "${TRAINING_DATA_DIR}"/* "${BACKUP_DIR}/" 2>/dev/null || true
        echo -e "${GREEN}Backup complete.${NC}"
    fi

# Create directories
mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${KHANA_DIR}"

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}gdown not found. Attempting to install...${NC}"
    python3 -m pip install --break-system-packages gdown 2>/dev/null || \
    python3 -m pip install --user gdown 2>/dev/null || \
    pip install gdown 2>/dev/null || {
        echo -e "${RED}Failed to install gdown. Please install manually: pip install gdown${NC}"
        exit 1
    }
fi

# Extract file ID if URL is provided
if [[ "$FILE_ID_OR_URL" == *"drive.google.com"* ]]; then
    # Extract file ID from URL
    FILE_ID=$(echo "$FILE_ID_OR_URL" | sed -n 's/.*\/d\/\([a-zA-Z0-9_-]*\).*/\1/p')
    if [ -z "$FILE_ID" ]; then
        echo -e "${RED}Error: Could not extract file ID from URL${NC}"
        exit 1
    fi
    echo -e "${GREEN}Extracted file ID: ${FILE_ID}${NC}"
else
    FILE_ID="$FILE_ID_OR_URL"
fi

# Download the dataset
echo -e "${GREEN}Downloading Khana dataset from Google Drive...${NC}"
echo -e "${YELLOW}This may take a while depending on file size...${NC}"

gdown --fuzzy "https://drive.google.com/file/d/${FILE_ID}/view" -O "${ZIP_FILE}" || {
    echo -e "${RED}Error: Download failed. Please check:${NC}"
    echo "  1. The file ID is correct"
    echo "  2. The file is publicly accessible or you have access"
    echo "  3. Your internet connection is working"
    exit 1
}

if [ ! -f "${ZIP_FILE}" ]; then
    echo -e "${RED}Error: Download failed. ZIP file not found.${NC}"
    exit 1
fi

echo -e "${GREEN}Download complete. File size: $(du -h ${ZIP_FILE} | cut -f1)${NC}"

# Extract the zip file
echo -e "${GREEN}Extracting dataset...${NC}"
unzip -q "${ZIP_FILE}" -d "${KHANA_DIR}" || {
    echo -e "${RED}Error: Failed to extract zip file. Please check if the file is valid.${NC}"
    exit 1
}

echo -e "${GREEN}Extraction complete.${NC}"

# Check the structure of extracted files
echo -e "${GREEN}Checking dataset structure...${NC}"
echo -e "${YELLOW}Contents of extracted dataset:${NC}"
ls -la "${KHANA_DIR}" | head -20

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Dataset location: ${KHANA_DIR}${NC}"
if [ -d "${BACKUP_DIR}" ]; then
    echo -e "${YELLOW}Old dataset backed up to: ${BACKUP_DIR}${NC}"
fi
echo ""
echo -e "${YELLOW}Next step: Run the setup script to organize the dataset${NC}"
echo "  python3 scripts/setup_khana_dataset.py"
echo ""
echo -e "${GREEN}Note: Khana dataset is in classification format (class folders)${NC}"
echo "The setup script will organize it into train/val/test structure."

