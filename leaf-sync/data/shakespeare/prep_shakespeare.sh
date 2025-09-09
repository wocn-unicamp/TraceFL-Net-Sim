#!/usr/bin/env bash
# prep_shakespeare_leaf.sh
# Pipeline completo do dataset Shakespeare (LEAF) usando o ZIP antigo 1994-01-100.zip.
# - Limpeza "como recomenda o LEAF": preserva raw_data/ se existir
# - Download idempotente do ZIP antigo (só baixa se faltar raw_data.txt)
# - Ao final, mostra quantos usuários e amostras ficaram no split de train

set -euo pipefail

ROOT="${1:-$HOME/github/TraceFL-Net-Sim/leaf-sync/data/shakespeare}"
echo "=> Raiz do pipeline: ${ROOT}"
cd "$ROOT"

have() { command -v "$1" >/dev/null 2>&1; }

json_len() {
  # $1 = caminho JSON; imprime len(obj) (ou 0 se ausente)
  local f="$1"
  if [[ ! -s "$f" ]]; then echo 0; return; fi
  if have jq; then jq '. | length' "$f"; else
    python3 - "$f" <<'PY'
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as fh:
    print(len(json.load(fh)))
PY
  fi
}

json_users_len() {
  # $1 = all_data.json; imprime len(.users)
  local f="$1"
  if [[ ! -s "$f" ]]; then echo 0; return; fi
  if have jq; then jq '.users | length' "$f"; else
    python3 - "$f" <<'PY'
import json,sys
with open(sys.argv[1], 'r', encoding='utf-8', errors='ignore') as fh:
    j=json.load(fh)
print(len(j.get("users", [])))
PY
  fi
}

# ===== Limpeza recomendada pelo LEAF: preservar raw_data/, limpar derivados =====
echo "=> Limpando artefatos derivados (preservando raw_data/ se existir)..."
rm -rf data/all_data data/train data/test data/sampled_data data/rem_user_data meta
mkdir -p data/all_data preprocess
# garanta existência de raw_data/ (sem apagar conteúdo existente)
mkdir -p data/raw_data

# ===== Download idempotente do ZIP antigo (só se faltar raw_data.txt) =====
RAW_TXT="data/raw_data/raw_data.txt"
if [[ ! -s "${RAW_TXT}" ]]; then
  echo "=> raw_data.txt não encontrado — baixando ZIP antigo 1994-01-100.zip..."
  URLS=(
    "https://mirror.ossplanet.net/mirror/gutenberg/1/0/100/old/1994-01-100.zip"
    "https://readingroo.ms/1/0/100/old/old/1994-01-100.zip"
  )
  pushd preprocess >/dev/null
    ok=false
    for u in "${URLS[@]}"; do
      echo "   - tentando: $u"
      if curl -fL --retry 3 --retry-delay 2 "$u" -o 1994-01-100.zip; then
        ok=true; break
      fi
    done
    if ! $ok; then
      echo "[ERRO] não foi possível baixar 1994-01-100.zip" >&2
      exit 1
    fi

    echo "=> Extraindo 100.txt..."
    unzip -o 1994-01-100.zip >/dev/null
    rm -f 1994-01-100.zip
    mv -f 100.txt ../data/raw_data/raw_data.txt
  popd >/dev/null
else
  echo "=> Encontrado ${RAW_TXT} — pulando download."
fi

# ===== Pré-processamento canônico do LEAF =====
echo "=> Gerando by_play_and_character/ e users_and_plays.json (parser original do LEAF)..."
python3 preprocess/preprocess_shakespeare.py data/raw_data/raw_data.txt data/raw_data/

echo "=> Empacotando all_data.json..."
python3 preprocess/gen_all_data.py

echo "=> Checando contadores (alvo ~1129):"
BY_PLAY=$( (ls -1 data/raw_data/by_play_and_character | wc -l) || echo 0 )
UP_LEN=$(json_len "data/raw_data/users_and_plays.json")
ALL_USERS=$(json_users_len "data/all_data/all_data.json")
printf "   - by_play_and_character: %s\n" "$BY_PLAY"
printf "   - users_and_plays.json : %s\n" "$UP_LEN"
printf "   - all_data.json (users): %s\n" "$ALL_USERS"

# ===== Split e estatísticas =====
echo "=> Split 80/20 (niid) e estatísticas..."
# use aqui os parâmetros que você desejar; abaixo deixo o exemplo que você tinha:
./preprocess.sh -s niid --sf 0.05 -k 64 -tf 0.9 -t sample
./stats.sh

# ===== Verificação final: quantos usuários e amostras ficaram no TRAIN =====
echo "=> Verificando conjunto de TREINO (arquivo mais recente em data/train/ *_train_*.json)..."
TRAIN_JSON="$(ls -1t data/train/*_train_*.json 2>/dev/null | head -n1 || true)"
if [[ -n "${TRAIN_JSON}" && -s "${TRAIN_JSON}" ]]; then
  echo "   - Train JSON: ${TRAIN_JSON}"
  if have jq; then
    TRAIN_USERS=$(jq '.users | length' "${TRAIN_JSON}")
    TRAIN_SAMPLES=$(jq 'reduce (.user_data[] | length) as $n (0; . + $n)' "${TRAIN_JSON}")
  else
    TRAIN_USERS=$(python3 - "${TRAIN_JSON}" <<'PY'
import json,sys
j=json.load(open(sys.argv[1], 'r', encoding='utf-8', errors='ignore'))
print(len(j.get("users", [])))
PY
)
    TRAIN_SAMPLES=$(python3 - "${TRAIN_JSON}" <<'PY'
import json,sys
j=json.load(open(sys.argv[1], 'r', encoding='utf-8', errors='ignore'))
print(sum(len(v) for v in j.get("user_data", {}).values()))
PY
)
  fi
  echo "   - Usuários no TRAIN : ${TRAIN_USERS}"
  echo "   - Amostras no TRAIN : ${TRAIN_SAMPLES}"
else
  echo "   - [WARN] não encontrei JSON de treino em data/train/"
fi

echo "=> Concluído."
