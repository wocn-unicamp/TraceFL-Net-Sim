#!/usr/bin/env bash
set -euo pipefail

# Argumentos:
#   $1 = carpeta de metadatos (por defecto: ./baseline)
#   $2 = carpeta de métricas   (por defecto: ./results)
output_dir="${1:-./baseline}"     # metadatos
results_dir="${2:-./results}"     # métricas (tendrá subcarpetas sys/ y stat/)

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="1000"

fedavg_lr="0.004"
# declare -a fedavg_vals=( "30 1" "20 1" "10 1" "5 1" "3 1") # (num_clients num_epochs)
declare -a fedavg_vals=( "64 2") # (num_clients num_epochs)


# minibatch_lr="0.06"
minibatch_lr="0.004"
# declare -a minibatch_vals=("30 0.9" "30 0.8"  "30 0.6" "30 0.5" "30 0.4" "30 0.2" "30 0.1") # (num_clients minibatch_fraction)
declare -a minibatch_vals=("20 1") # (num_clients minibatch_fraction)

###################### Functions ###################################

# Mueve un archivo si existe y lo renombra con sufijo
# $1 = origen (relativo a models/metrics)
# $2 = destino (results_dir/<sys|stat>)
# $3 = prefijo de salida (sys_metrics | stat_metrics)
# $4 = sufijo (ej.: fedavg_c_3_e_1)
# Move um arquivo de métricas (se existir) para results/<sys|stat>/ com sobrescrita
function _move_one() {
  local src="$1"
  local dst_dir="$2"
  local out_prefix="$3"
  local suffix="$4"

  if [[ -f "$src" ]]; then
    mkdir -p "${dst_dir}"
    mv -f "$src" "${dst_dir}/${out_prefix}_${suffix}.csv"
  else
    echo "WARN: não encontrado: '$src' (omitindo)."
  fi
}

# Copia métricas e metadados; cria meta_<sufixo> do zero e apaga se já existir
function move_data() {
  local meta_root="$1"     # ex.: ./baseline (absoluto depois do realpath)
  local metrics_root="$2"  # ex.: ./results  (absoluto)
  local suffix="$3"        # ex.: fedavg_c_50_e_1

  # Pastas de saída
  mkdir -p "${metrics_root}/sys" "${metrics_root}/stat" "${meta_root}"

  # Métricas -> results/sys e results/stat
  pushd models/metrics >/dev/null
    _move_one "metrics_sys.csv"  "${metrics_root}/sys"  "sys_metrics"  "${suffix}"
    _move_one "metrics_stat.csv" "${metrics_root}/stat" "stat_metrics" "${suffix}"
  popd >/dev/null

  # Metadados -> baseline/meta_<sufixo>
  if [[ -d "data/femnist/meta" ]]; then
    local target="${meta_root}/meta_${suffix}"
    rm -rf "${target}"                 # evita "Directory not empty"
    mkdir -p "${target}"
    # copia o conteúdo de meta/ para o destino (sem criar diretório meta/ dentro)
    cp -a "data/femnist/meta/." "${target}/"
  else
    echo "WARN: não existe data/femnist/meta; metadados não copiados."
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

# Run FedAvg experiments
# for val_pair in "${fedavg_vals[@]}"; do
#   clients_per_round="$(echo ${val_pair} | cut -d' ' -f1)"
#   num_epochs="$(echo ${val_pair} | cut -d' ' -f2)"
#   echo "Running FedAvg: epochs=${num_epochs}, clients=${clients_per_round}"
#   run_fedavg "${clients_per_round}" "${num_epochs}"
# done

popd >/dev/null
