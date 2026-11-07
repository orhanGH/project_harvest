#!/bin/bash
set -euo pipefail

REPO_DIR=${1:-"$HOME/project_harvest/HARVEST"}
ENV_NAME=${HARVEST_ENV_NAME:-harvest-tesseract}
TESSDATA_DIR=${HARVEST_TESSDATA_DIR:-"$HOME/.local/share/harvest-tessdata"}
TESSDATA_BASE_URL=${HARVEST_TESSDATA_BASE_URL:-https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main}

module purge
module load Miniforge3
eval "$(conda shell.bash hook)"
cd "$REPO_DIR"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$REPO_DIR/environments/tesseract.yml" --prune
else
    conda env create -n "$ENV_NAME" -f environments/tesseract.yml
fi

conda activate "$ENV_NAME"
python -m pip install -e "${REPO_DIR}[base]"

mkdir -p "$TESSDATA_DIR"
for language in eng fra; do
    target="$TESSDATA_DIR/${language}.traineddata"
    if [ ! -s "$target" ]; then
        temporary="${target}.part"
        curl --fail --location --retry 3 \
            --output "$temporary" \
            "$TESSDATA_BASE_URL/${language}.traineddata"
        mv "$temporary" "$target"
    fi
done
export TESSDATA_PREFIX="$TESSDATA_DIR"

python -c "import cv2, fitz, pytesseract, yaml; print('Python dependencies: OK')"
tesseract --version | head -n 1
tesseract --list-langs
python - <<'PY'
import pytesseract

available = set(pytesseract.get_languages(config=""))
missing = {"eng", "fra"} - available
if missing:
    raise SystemExit(f"Missing Tesseract languages: {sorted(missing)}")
print("Tesseract languages: eng + fra OK")
PY
