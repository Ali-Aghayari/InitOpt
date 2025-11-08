#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <GOOGLE_DRIVE_FOLDER_ID> [output_zip_name]"
  echo "Example: $0 1Z0wg4HADhpgrztyT3eWijPbJJN5Y2jQt myfolder.zip"
  exit 1
fi

FOLDER_ID="$1"
ZIP_NAME="${2:-gdrive_folder.zip}"

# Pick a Python
PYBIN="${PYBIN:-python3}"
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYBIN="python"
  else
    echo "[!] No python interpreter found (python3 or python)."
    exit 2
  fi
fi

# Make sure pip exists; try to bootstrap if missing (no sudo)
if ! "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "[*] Bootstrapping pip locally..."
  "$PYBIN" -m ensurepip --user >/dev/null 2>&1 || true
fi

# Ensure ~/.local/bin is on PATH for this session
export PATH="$PATH:$HOME/.local/bin"

# Install gdown locally if missing
if ! command -v gdown >/dev/null 2>&1; then
  echo "[*] Installing gdown locally (no sudo)..."
  "$PYBIN" -m pip install --user --quiet --upgrade pip >/dev/null 2>&1 || true
  "$PYBIN" -m pip install --user --quiet gdown
  hash -r || true
fi

# Re-check gdown
if ! command -v gdown >/dev/null 2>&1; then
  echo "[!] gdown not found and could not be installed. Make sure Python + pip are available."
  exit 3
fi

# Workspace
WORKDIR="$(mktemp -d)"
DL_DIR="$WORKDIR/download"
mkdir -p "$DL_DIR"

# Clean up temp dir on exit
cleanup() {
  rm -rf "$WORKDIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "[*] Downloading ALL files from folder ID: $FOLDER_ID"
# --remaining-ok is crucial for folders with >50 files
gdown --folder --remaining-ok --id "$FOLDER_ID" -O "$DL_DIR"

echo "[*] Zipping contents into: $ZIP_NAME (using Python zipfile)"
# Export so the Python snippet can read them
export DL_DIR ZIP_NAME

"$PYBIN" - <<'EOF'
import os, sys, zipfile

dl_dir = os.environ["DL_DIR"]
zip_name = os.environ["ZIP_NAME"]

# Create zip at the current working directory
abs_zip = os.path.abspath(zip_name)

with zipfile.ZipFile(abs_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(dl_dir):
        for f in files:
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, dl_dir)  # keep folder structure
            zf.write(abs_path, rel_path)

print(f"[✓] Done. Output: {abs_zip}")
EOF

echo "[✓] Finished."
