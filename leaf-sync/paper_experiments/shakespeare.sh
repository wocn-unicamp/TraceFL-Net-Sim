#!/usr/bin/env bash
set -euo pipefail

# ============ Descoberta de caminhos (independente do diretório atual) ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"         # .../leaf-sync/paper_experiments
LEAFSYNC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                     # .../leaf-sync
ROOT_DIR="$(cd "${LEAFSYNC_DIR}/.." && pwd)"                       # .../TraceFL-Net-Sim

MODELS_DIR="${LEAFSYNC_DIR}/models"
DATASET_DIR="${LEAFSYNC_DIR}/data/shakespeare"
UTILS_DIR="${LEAFSYNC_DIR}/data/utils"
METRICS_DIR="${MODELS_DIR}/metrics"

# Tudo dentro do próprio leaf-sync
LEAF_DATA_DIR="${DATASET_DIR}"                 # => .../leaf-sync/data/shakespeare
PREPROCESS_DIR="${LEAF_DATA_DIR}/preprocess"   # => .../leaf-sync/data/shakespeare/preprocess

# Argumentos:
#   $1 = carpeta de metadatos (default: leaf-sync/baseline)
#   $2 = carpeta de métricas  (default: leaf-sync/results)
output_dir="${1:-${LEAFSYNC_DIR}/baseline}"     # metadatos
results_dir="${2:-${LEAFSYNC_DIR}/results}"     # métricas (tendrá subcarpetas sys/ y stat/)


split_seed="1549786796"
sampling_seed="1549786595"


num_rounds="50"
fedavg_lr="0.8"

declare -a fedavg_vals=("8 1") # (num_clients num_epochs)

echo "[Paths]"
echo "  ROOT_DIR:      ${ROOT_DIR}"
echo "  LEAFSYNC_DIR:  ${LEAFSYNC_DIR}"
echo "  LEAF_DATA_DIR: ${LEAF_DATA_DIR}"
echo "  MODELS_DIR:    ${MODELS_DIR}"
echo "  DATASET_DIR:   ${DATASET_DIR}"
echo "  UTILS_DIR:     ${UTILS_DIR}"
echo

###################### Helpers ###################################
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

###################### Pré-processamento (LEAF oficial) ##########################
# Usa ZIP antigo (1994-01-100.zip). Limpa apenas artefatos derivados (preserva raw_data/).
# Depois do split, verifica quantos usuários e amostras ficaram no TRAIN.
prep_shakespeare_preprocess() {
  echo "==[prep] Iniciando pré-processamento DENTRO do leaf-sync =="
  echo "   LEAF_DATA_DIR (=DATASET_DIR): ${LEAF_DATA_DIR}"
  echo "   PREPROCESS_DIR              : ${PREPROCESS_DIR}"

  mkdir -p "${LEAF_DATA_DIR}"
  pushd "${LEAF_DATA_DIR}" >/dev/null
    # 0) Limpeza (preserva raw_data/)
    echo "==[prep] Limpando derivados (preservando raw_data/)..."
    rm -rf data/all_data data/train data/test data/sampled_data data/rem_user_data meta
    mkdir -p data/raw_data data/all_data

    # 1) Download idempotente do ZIP antigo (só se faltar raw_data.txt)
    RAW_TXT="data/raw_data/raw_data.txt"
    if [[ ! -s "${RAW_TXT}" ]]; then
      echo "==[prep] Baixando 1994-01-100.zip..."
      mkdir -p preprocess_tmp
      pushd preprocess_tmp >/dev/null
        ZIP_URLS=(
          "https://mirror.ossplanet.net/mirror/gutenberg/1/0/100/old/1994-01-100.zip"
          "https://readingroo.ms/1/0/100/old/old/1994-01-100.zip"
        )
        ok=false
        for u in "${ZIP_URLS[@]}"; do
          echo "   - tentando: $u"
          if curl -fL --retry 3 --retry-delay 2 "$u" -o 1994-01-100.zip; then ok=true; break; fi
        done
        if ! $ok; then echo "!! Falha no download." >&2; exit 1; fi
        unzip -o 1994-01-100.zip >/dev/null
        rm -f 1994-01-100.zip
        mv -f 100.txt ../data/raw_data/raw_data.txt
      popd >/dev/null
      rm -rf preprocess_tmp
    else
      echo "==[prep] Encontrado ${RAW_TXT} — pulando download."
    fi

    # 2) Pré-processamento (usando scripts do próprio leaf-sync)
    echo "==[prep] Rodando preprocess_shakespeare.py + gen_all_data.py (locais)..."
    python3 "${PREPROCESS_DIR}/preprocess_shakespeare.py" data/raw_data/raw_data.txt data/raw_data/
    python3 "${PREPROCESS_DIR}/gen_all_data.py"

    # 3) Sanidade
    echo "==[prep] Checando contadores:"
    BY_PLAY=$( (ls -1 data/raw_data/by_play_and_character | wc -l) || echo 0 )
    USERS_PLAYS=$(json_len "data/raw_data/users_and_plays.json")
    ALLDATA_USERS=$(json_users_len "data/all_data/all_data.json")
    echo "   - by_play_and_character: ${BY_PLAY}"
    echo "   - users_and_plays.json : ${USERS_PLAYS}"
    echo "   - all_data.json (users): ${ALLDATA_USERS}"

    # 4) Split (niid) e 5) Stats, ambos dentro do mesmo dir
    echo "==[prep] Split (niid)..."
    bash ./preprocess.sh -s niid --sf 0.05 -k 64 -tf 0.9 -t sample --smplseed "${sampling_seed}" --spltseed "${split_seed}"

    bash ./stats.sh

    # 6) Verificação final do TRAIN
    echo "==[prep] Verificando JSON de TREINO..."
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

    # 7) Nada de cópia: já estamos no DATASET_DIR/LEAF_DATA_DIR
    echo "==[prep] Concluído. Dataset pronto em ${LEAF_DATA_DIR}"
  popd >/dev/null
}


###################### Treino ###################################

function run_fedavg() {
  local clients_per_round="$1"
  local num_epochs="$2"

  pushd "${MODELS_DIR}" >/dev/null
    python main.py -dataset 'shakespeare' -model 'stacked_lstm' \
      --num-rounds "${num_rounds}" \
      --clients-per-round "${clients_per_round}" \
      --num-epochs "${num_epochs}" \
      --eval-every 2 \
      -lr "${fedavg_lr}"
  popd >/dev/null

  move_data "${output_dir}" "${results_dir}" "shakespeare_fedavg_c_${clients_per_round}_e_${num_epochs}"
}

function run_minibatch() {
  local clients_per_round="$1"
  local minibatch_percentage="$2"

  pushd "${MODELS_DIR}" >/dev/null
    python main.py -dataset 'shakespeare' -model 'stacked_lstm' \
      --minibatch "${minibatch_percentage}" \
      --num-rounds "${num_rounds}" \
      --clients-per-round "${clients_per_round}" \
      -lr "${minibatch_lr}"
  popd >/dev/null

  move_data "${output_dir}" "${results_dir}" "shakespeare_minibatch_c_${clients_per_round}_mb_${minibatch_percentage}"
}

# Mueve un arquivo se existe e renomeia com sufixo
function _move_one() {
  local src="$1"
  local dst_dir="$2"
  local out_prefix="$3"
  local suffix="$4"

  if [[ -f "$src" ]]; then
    mv "$src" "${dst_dir}/${out_prefix}_${suffix}.csv"
  else
    echo "WARN: no se encontró '$src' (se omite)."
  fi
}

# Guarda métricas em results_dir e meta em output_dir
function move_data() {
  local meta_path="$1"     # baseline
  local metrics_path="$2"  # results
  local suffix="$3"

  mkdir -p "${meta_path}"
  mkdir -p "${metrics_path}/sys" "${metrics_path}/stat"

  pushd "${METRICS_DIR}" >/dev/null
    _move_one "metrics_sys.csv"  "${metrics_path}/sys"  "sys_metrics"  "${suffix}"
    _move_one "metrics_stat.csv" "${metrics_path}/stat" "stat_metrics" "${suffix}"
  popd >/dev/null

  if [[ -d "${DATASET_DIR}/meta" ]]; then
    cp -r "${DATASET_DIR}/meta" "${meta_path}" || true
    if [[ -d "${meta_path}/meta" ]]; then
      mv "${meta_path}/meta" "${meta_path}/meta_${suffix}"
    fi
  else
    echo "WARN: no existe ${DATASET_DIR}/meta; se omite copia de meta."
  fi
}

##################### Script #################################

# Check that data and models are available
if [[ ! -d "${LEAFSYNC_DIR}/data" || ! -d "${MODELS_DIR}" ]]; then
  echo "Couldn't find ${LEAFSYNC_DIR}/data and/or ${MODELS_DIR} - verifique a árvore do repo"
fi

# Crear carpetas y normalizar rutas
mkdir -p "${output_dir}" "${results_dir}/sys" "${results_dir}/stat"
output_dir="$(realpath "${output_dir}")"
results_dir="$(realpath "${results_dir}")"
echo "Metadatos en: ${output_dir}"
echo "Métricas SYS en:  ${results_dir}/sys"
echo "Métricas STAT en: ${results_dir}/stat"
echo "Invoca: ${0} <dir_metadatos> <dir_metricas>  para cambiar rutas"
echo

# === Pré-processamento Shakespeare (LEAF oficial + cópia para leaf-sync) ===
prep_shakespeare_preprocess

# # Run minibatch SGD experiments (opcional)
# for val_pair in "${minibatch_vals[@]}"; do
#   clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
#   minibatch_percentage="$(echo ${val_pair} | cut -d' ' -f2)"
#   echo "Running Minibatch experiment with fraction ${minibatch_percentage} and ${clients_per_round}"
#   run_minibatch "${clients_per_round}" "${minibatch_percentage}"
# done

# Run FedAvg experiments
for val_pair in "${fedavg_vals[@]}"; do
  clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
  num_epochs="$(echo ${val_pair} | cut -d' ' -f2)"
  echo "Running FedAvg: epochs=${num_epochs}, clients=${clients_per_round}"
  run_fedavg "${clients_per_round}" "${num_epochs}"
done
