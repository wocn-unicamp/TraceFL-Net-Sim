#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${SHK_DIR}/data/raw_data"
OUT_FILE="${RAW_DIR}/raw_data.txt"

mkdir -p "${RAW_DIR}"

if command -v curl >/dev/null 2>&1; then
  DOWNLOAD() { curl -fL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD() { wget -qO "$2" "$1"; }
else
  echo "[ERRO] Nem 'curl' nem 'wget' encontrados." >&2
  exit 1
fi

# Coloque 'pg100.txt' primeiro (mais compatível com o parser original)
URLS=(
  "https://www.gutenberg.org/cache/epub/100/pg100.txt"
  "https://www.gutenberg.org/files/100/100-0.txt"
)

echo ">>> Baixando Shakespeare (Project Gutenberg) para: ${OUT_FILE}"
rm -f "${OUT_FILE}"

ok=false
for url in "${URLS[@]}"; do
  echo "Tentando: ${url}"
  if DOWNLOAD "${url}" "${OUT_FILE}"; then
    if [[ -s "${OUT_FILE}" ]]; then
      ok=true
      echo "OK: download concluído."
      break
    fi
  fi
done

if [[ "${ok}" != true ]]; then
  echo "[ERRO] Não foi possível obter o texto de Shakespeare." >&2
  exit 1
fi

echo "Arquivo pronto: ${OUT_FILE}"
