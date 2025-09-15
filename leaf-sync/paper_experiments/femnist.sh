#!/usr/bin/env bash
set -euo pipefail

# Argumentos:
#   $1 = carpeta de metadatos (por defecto: ./baseline)
#   $2 = carpeta de métricas   (por defecto: ./results)
output_dir="${1:-./baseline}"     # metadatos
results_dir="${2:-./results}"     # métricas (tendrá subcarpetas sys/ y stat/)

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="500"
eval_every="20"



fedavg_lr="0.004"
# declare -a fedavg_vals=( "5 1" "10 1" "30 1" "50 1" ) # (num_clients num_epochs)
# declare -a fedavg_vals=( "5 1" ) # (num_clients num_epochs)
declare -a fedavg_vals=( "10 1" "30 1" "50 1" ) # (num_clients num_epochs)


minibatch_lr="0.06"
# declare -a minibatch_vals=( "3 1" "3 0.1" "5 1" ) # (num_clients minibatch_fraction)
declare -a minibatch_vals=("10 1" "30 1" "50 1") # (num_clients minibatch_fraction)
###################### Functions ###################################

# Mueve un archivo si existe y lo renombra con sufijo
# $1 = origen (relativo a models/metrics)
# $2 = destino (results_dir/<sys|stat>)
# $3 = prefijo de salida (sys_metrics | stat_metrics)
# $4 = sufijo (ej.: fedavg_c_3_e_1)
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
  pushd models/metrics >/dev/null
    # En tu repo se llaman así:
    #   metrics_sys.csv  -> sys_metrics_<sufijo>.csv  (va a results/sys/)
    #   metrics_stat.csv -> stat_metrics_<sufijo>.csv (va a results/stat/)
    _move_one "metrics_sys.csv"  "${metrics_path}/sys"  "sys_metrics"  "${suffix}"
    _move_one "metrics_stat.csv" "${metrics_path}/stat" "stat_metrics" "${suffix}"
  popd >/dev/null

  # Metadatos -> baseline/
  if [[ -d "data/femnist/meta" ]]; then
    cp -r "data/femnist/meta" "${meta_path}" || true
    if [[ -d "${meta_path}/meta" ]]; then
      mv "${meta_path}/meta" "${meta_path}/meta_${suffix}"
    fi
  else
    echo "WARN: no existe data/femnist/meta; se omite copia de meta."
  fi
}

function run_fedavg() {
  local clients_per_round="$1"
  local num_epochs="$2"

  pushd models/ >/dev/null
    python main.py -dataset 'femnist' -model 'cnn' \
      --num-rounds "${num_rounds}" \
      --clients-per-round "${clients_per_round}" \
      --num-epochs "${num_epochs}" \
      --eval-every "${eval_every}" \
      -lr "${fedavg_lr}"
  popd >/dev/null

  move_data "${output_dir}" "${results_dir}" "fedavg_c_${clients_per_round}_e_${num_epochs}"
}

function run_minibatch() {
  local clients_per_round="$1"
  local minibatch_percentage="$2"

  pushd models/ >/dev/null
    python main.py -dataset 'femnist' -model 'cnn' \
      --minibatch "${minibatch_percentage}" \
      --num-rounds "${num_rounds}" \
      --clients-per-round "${clients_per_round}" \
      --eval-every "${eval_every}" \
      -lr "${minibatch_lr}"
  popd >/dev/null

  move_data "${output_dir}" "${results_dir}" "minibatch_c_${clients_per_round}_mb_${minibatch_percentage}"
}

##################### Script #################################
pushd ../ >/dev/null

# Check that data and models are available
if [[ ! -d 'data/' || ! -d 'models/' ]]; then
  echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# If data unavailable, execute pre-processing script
if [[ ! -d 'data/femnist/data/train' ]]; then
  if [[ ! -f 'data/femnist/preprocess.sh' ]]; then
    echo "Couldn't find data/femnist/preprocess.sh - get https://github.com/TalwalkarLab/leaf"
    exit 1
  fi
  echo "Couldn't find FEMNIST data - running data preprocessing script"
  pushd data/femnist/ >/dev/null
    rm -rf meta/ data/test data/train data/rem_user_data data/intermediate
    ./preprocess.sh -s niid --sf 0.05 -k 100 -t sample --smplseed "${sampling_seed}" --spltseed "${split_seed}"
  popd >/dev/null
fi

# Crear carpetas y normalizar rutas
mkdir -p "${output_dir}" "${results_dir}/sys" "${results_dir}/stat"
output_dir="$(realpath "${output_dir}")"
results_dir="$(realpath "${results_dir}")"
echo "Metadatos en: ${output_dir}"
echo "Métricas SYS en:  ${results_dir}/sys"
echo "Métricas STAT en: ${results_dir}/stat"
echo "Invoca: ${0} <dir_metadatos> <dir_metricas>  para cambiar rutas"

# Run minibatch SGD experiments
for val_pair in "${minibatch_vals[@]}"; do
  clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
  minibatch_percentage="$(echo ${val_pair} | cut -d' ' -f2)"
  echo "Running Minibatch experiment with fraction ${minibatch_percentage} and ${clients_per_round} clients"
  run_minibatch "${clients_per_round}" "${minibatch_percentage}"
done

# # Run FedAvg experiments
# for val_pair in "${fedavg_vals[@]}"; do
#   clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
#   num_epochs="$(echo ${val_pair} | cut -d' ' -f2)"
#   echo "Running FedAvg: epochs=${num_epochs}, clients=${clients_per_round}"
#   run_fedavg "${clients_per_round}" "${num_epochs}"
# done

popd >/dev/null
