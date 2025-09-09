#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OUT_DIR="${SCRIPT_DIR}/../data/raw_data"
mkdir -p "${OUT_DIR}"

TARGET="${OUT_DIR}/raw_data.txt"

# Idempotência
if [[ -s "${TARGET}" ]]; then
  echo "[get_data] '${TARGET}' já existe — pulando download."
  exit 0
fi

# Preferir o ZIP antigo; fallback para TXT modernos
ZIP_URL="https://mirror.ossplanet.net/mirror/gutenberg/1/0/100/old/1994-01-100.zip"
FALLBACK_TXT_URLS=(
  "https://www.gutenberg.org/cache/epub/100/pg100.txt"
  "https://www.gutenberg.org/files/100/100-0.txt"
  "https://www.gutenberg.org/files/100/100.txt"
)

download_zip_and_extract() {
  local url="$1"
  echo "[get_data] tentando ZIP antigo: $url"
  local tmpzip="${OUT_DIR}/_tmp_1994-01-100.zip"
  rm -f "${tmpzip}"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "${tmpzip}" || return 1
  unzip -o "${tmpzip}" -d "${SCRIPT_DIR}" >/dev/null
  rm -f "${tmpzip}"
  mv -f "${SCRIPT_DIR}/100.txt" "${TARGET}"
  echo "[get_data] salvo em ${TARGET}"
  return 0
}

download_txt() {
  local url="$1"
  echo "[get_data] tentando TXT: $url"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "${TARGET}.part" || return 1
  mv "${TARGET}.part" "${TARGET}"
  echo "[get_data] salvo em ${TARGET}"
  return 0
}

# 1) ZIP antigo primeiro
if download_zip_and_extract "${ZIP_URL}"; then
  exit 0
fi

# 2) Fallbacks modernos
for u in "${FALLBACK_TXT_URLS[@]}"; do
  if download_txt "$u"; then
    exit 0
  fi
done

echo "[get_data] falha ao obter Shakespeare (#100)." >&2
exit 1
