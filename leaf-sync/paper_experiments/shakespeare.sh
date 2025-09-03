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

# Argumentos:
#   $1 = carpeta de metadatos (default: leaf-sync/baseline)
#   $2 = carpeta de métricas  (default: leaf-sync/results)
output_dir="${1:-${LEAFSYNC_DIR}/baseline}"     # metadatos
results_dir="${2:-${LEAFSYNC_DIR}/results}"     # métricas (tendrá subcarpetas sys/ y stat/)

# split_seed=""
# sampling_seed=""

# split_seed="1549786796"
# sampling_seed="1549786595"


split_seed="0"
sampling_seed="0"

num_rounds="50"

fedavg_lr="0.08"
# fedavg_lr="0.004"

declare -a fedavg_vals=("8 1") # (num_clients num_epochs)
# minibatch_lr="0.06"
# declare -a minibatch_vals=( "30 0.1" "30 0.2" "30 0.5" "30 0.8" )

echo "[Paths]"
echo "  ROOT_DIR:      ${ROOT_DIR}"
echo "  LEAFSYNC_DIR:  ${LEAFSYNC_DIR}"
echo "  MODELS_DIR:    ${MODELS_DIR}"
echo "  DATASET_DIR:   ${DATASET_DIR}"
echo "  UTILS_DIR:     ${UTILS_DIR}"
echo

###################### Functions ###################################

# Mueve un archivo si existe y lo renombra con sufijo
# $1 = origen (relativo a MODELS/metrics)
# $2 = destino (results_dir/<sys|stat>)
# $3 = prefijo de salida (sys_metrics | stat_metrics)
# $4 = sufijo (ej.: shakespeare_fedavg_c_5_e_1)
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

# Guarda métricas en results_dir (separadas en sys/ y stat/) y meta en output_dir
function move_data() {
  local meta_path="$1"     # baseline
  local metrics_path="$2"  # results
  local suffix="$3"

  # Crear carpetas de salida
  mkdir -p "${meta_path}"
  mkdir -p "${metrics_path}/sys"   # métricas del sistema
  mkdir -p "${metrics_path}/stat"  # métricas estadísticas

  # Métricas -> results/sys/ y results/stat/
  pushd "${METRICS_DIR}" >/dev/null
    _move_one "metrics_sys.csv"  "${metrics_path}/sys"  "sys_metrics"  "${suffix}"
    _move_one "metrics_stat.csv" "${metrics_path}/stat" "stat_metrics" "${suffix}"
  popd >/dev/null

  # Metadatos -> baseline/ (provenientes de leaf-sync/data/shakespeare/meta)
  if [[ -d "${DATASET_DIR}/meta" ]]; then
    cp -r "${DATASET_DIR}/meta" "${meta_path}" || true
    if [[ -d "${meta_path}/meta" ]]; then
      mv "${meta_path}/meta" "${meta_path}/meta_${suffix}"
    fi
  else
    echo "WARN: no existe ${DATASET_DIR}/meta; se omite copia de meta."
  fi
}

# Preprocess SIEMPRE usando leaf-sync/data/utils/preprocess.sh
function preprocess_shakespeare() {
  if [[ ! -f "${UTILS_DIR}/preprocess.sh" ]]; then
    echo "Couldn't find ${UTILS_DIR}/preprocess.sh"
    echo "Asegúrate de estar en la raíz correcta del repo (TraceFL-Net-Sim tiene leaf-sync/data/utils/preprocess.sh)"
    exit 1
  fi

  echo "[Preprocess] Limpiando ${DATASET_DIR}/{meta, data/...} y regenerando dataset (train/test)"
    rm -rf "${DATASET_DIR}/meta" \
        "${DATASET_DIR}/data/all_data" \
        "${DATASET_DIR}/data/test" \
        "${DATASET_DIR}/data/train" \
        "${DATASET_DIR}/data/rem_user_data" \
        "${DATASET_DIR}/data/intermediate" \
        "${DATASET_DIR}/data/sampled_data"
  # Execução exatamente como você indicou:
  pushd "${UTILS_DIR}" >/dev/null
    # bash -x ./preprocess.sh --name shakespeare -s niid --sf 1.0 -k 0 -tf 0.8 -t sample --smplseed "${sampling_seed}" --spltseed "${split_seed}"
    bash -x ./preprocess.sh --name shakespeare -s niid --sf 0.05 -k 64 -tf 0.9 -t sample  --smplseed "${sampling_seed}" --spltseed "${split_seed}" # same configuration as Leaf paper 
  popd >/dev/null

  # Checagem rápida
  if [[ ! -d "${DATASET_DIR}/data/train" || ! -d "${DATASET_DIR}/data/test" ]]; then
    echo "ERROR: preprocess terminou mas train/ ou test/ não foram criados em ${DATASET_DIR}/data"
    exit 1
  fi
}

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

# SIEMPRE executar o preprocess de Shakespeare (gera train/ e test/)
preprocess_shakespeare

# # Run minibatch SGD experiments
# for val_pair in "${minibatch_vals[@]}"; do
#   clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
#   minibatch_percentage="$(echo ${val_pair} | cut -d' ' -f2)"
#   echo "Running Minibatch experiment with fraction ${minibatch_percentage} and ${clients_per_round} clients"
#   run_minibatch "${clients_per_round}" "${minibatch_percentage}"
# done

# Run FedAvg experiments
for val_pair in "${fedavg_vals[@]}"; do
  clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
  num_epochs="$(echo ${val_pair} | cut -d' ' -f2)"
  echo "Running FedAvg: epochs=${num_epochs}, clients=${clients_per_round}"
  run_fedavg "${clients_per_round}" "${num_epochs}"
done
