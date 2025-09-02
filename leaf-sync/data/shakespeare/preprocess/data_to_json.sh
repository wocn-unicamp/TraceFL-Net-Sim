#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p ../data ../data/raw_data

# Garante o raw (usa o get_data.sh atualizado)
if [ ! -f ../data/raw_data/raw_data.txt ]; then
  ./get_data.sh
fi

# Executa o parser se a pasta não existir OU estiver vazia
# roda o parser se a pasta não existir OU estiver vazia
if [ ! -d "../data/raw_data/by_play_and_character" ] || [ -z "$(ls -A ../data/raw_data/by_play_and_character 2>/dev/null || true)" ]; then
  echo "dividing txt data between users"
  python3 preprocess_shakespeare.py ../data/raw_data/raw_data.txt ../data/raw_data/
fi


# Propaga --raw se foi passado
RAWTAG=""
if [[ "$*" == *"--raw"* ]]; then
  RAWTAG="--raw"
fi

# Gera all_data.json apenas se não existir nada
mkdir -p ../data/all_data
if [ -z "$(ls -A ../data/all_data 2>/dev/null || true)" ]; then
  echo "generating all_data.json"
  python3 gen_all_data.py $RAWTAG
fi
