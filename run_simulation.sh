#!/bin/bash
set -euo pipefail
export LC_NUMERIC=C

# ==============================================================================
# CONFIGURAÇÕES DO SIMULADOR E AMBIENTE
# ==============================================================================
SIM_BIN="${SIM_BIN:-./trace_driven_simulator/sim_bin}"
MAX_PARALLEL_SEEDS=5

# Validação do simulador
if [ -f "${SIM_BIN}" ] && [ -x "${SIM_BIN}" ]; then
  echo "=> Utilizando simulador pré-compilado: ${SIM_BIN}"
  COMPILED_BY_SCRIPT=0
elif command -v go >/dev/null 2>&1; then
  echo "=> Binário não encontrado. Compilando com o compilador Go local..."
  go build -o "${SIM_BIN}" trace_driven_simulator/main.go
  COMPILED_BY_SCRIPT=1
else
  echo "----------------------------------------------------------------------" >&2
  echo "ERRO: O binário '${SIM_BIN}' não foi encontrado e o Go não está instalado." >&2
  echo "" >&2
  echo "Compile o simulador na sua máquina local com:" >&2
  echo "  go build -o sim_bin trace_driven_simulator/main.go" >&2
  echo "Depois, envie-o ao servidor e informe o caminho:" >&2
  echo "  SIM_BIN=/caminho/para/sim_bin ./seu_script.sh" >&2
  echo "----------------------------------------------------------------------" >&2
  exit 1
fi

PROJECT_ROOT="$(pwd)"
SIM_BIN_ABS="$(realpath "${SIM_BIN}")"
TMP_BASE="/tmp/fl_sim_${USER}"
mkdir -p "${TMP_BASE}"

# ==============================================================================
# LISTAS E PARÂMETROS
# ==============================================================================
datasets=("femnist" "shakespeare")
algorithms=("fedavg" "minibatch")
bg_models=("MULTI")
transmission_success_rate=(1.0)
scenarios=("heterogeneous")
nclients_femnist=(3 5 10 20 30 50)
nclients_shakespeare=(2 3 4 5 8 10 20)
minibatch_vals=(0.2 0.4 0.5 0.6 0.8 0.9 1)
flop_val=(500000000)
seeds=(42 1337 2026 8888 9999)

# Constantes de rede e hardware
clients_bwd=1500000000 # 1.5 Gbps
server_bwd=2250000000  # 2.25 Gbps
bg_workload=0.67
number_cores=4
p_femnist=0.95
p_shakespeare=0.4
early_stop_femnist=278
early_stop_shakespeare=-1

output_dir="trace_driven_simulator/data"

# ==============================================================================
# FUNÇÕES AUXILIARES E TRAPS
# ==============================================================================
cleanup() {
  echo -e "\n[!] Sinal de interrupção recebido. Encerrando jobs paralelos..."
  kill $(jobs -p) 2>/dev/null || true
  rm -rf "${TMP_BASE}"
  if [ "${COMPILED_BY_SCRIPT:-0}" -eq 1 ]; then
    rm -f "${SIM_BIN_ABS}"
  fi
  exit 1
}

trap cleanup SIGINT SIGTERM

amdahl_speedup() {
  local cores=$1
  local p=$2
  echo "scale=4; 1 / ((1 - $p) + ($p / $cores))" | bc -l
}

wait_for_slot() {
  while [ "$(jobs -r -p | wc -l)" -ge "${MAX_PARALLEL_SEEDS}" ]; do
    wait -n 2>/dev/null || sleep 0.2
  done
}

# ==============================================================================
# INÍCIO DO FLUXO
# ==============================================================================
echo "Starting simulation script..."

for flops in "${flop_val[@]}"; do
  echo "Processing with FLOPs: ${flops}..."

  for scen in "${scenarios[@]}"; do
    echo "===================================================="
    echo "Processing Scenario: ${scen}"
    echo "===================================================="
    flops_mode="${scen}"

    # --- 1. Pré-processamento de Dados ---
    for dataset in "${datasets[@]}"; do
      echo "Preprocessing data for dataset: ${dataset} [Scenario: ${scen}]..."

      if [ "${dataset}" == "shakespeare" ]; then
        p="${p_shakespeare}"
      else
        p="${p_femnist}"
      fi

      speedup=$(amdahl_speedup "${number_cores}" "${p}")

      if [ "${scen}" == "homogeneous" ]; then
        flops_adjusted=$(echo "${flops} * ${speedup}" | bc -l)
        flops_adjusted=$(printf "%.0f" "${flops_adjusted}")

        python3 trace_driven_simulator/data_processor.py \
          --sample-dir "traces/sys" \
          --search-pattern "sys_metrics_${dataset}_*" \
          --output-dir "${output_dir}/${scen}/${flops}/" \
          --flops-mode "${flops_mode}" \
          --clients-flops "${flops_adjusted}"
      else
        python3 trace_driven_simulator/data_processor.py \
          --sample-dir "traces/sys_bimodal_amdahl" \
          --search-pattern "sys_metrics_${dataset}_*" \
          --output-dir "${output_dir}/${scen}/${flops}/" \
          --flops-mode "${flops_mode}"
      fi
    done

    # --- 2. Execução das Simulações ---
    for dataset in "${datasets[@]}"; do
      echo "----------------------------------------------------"
      echo "Starting simulations for dataset: ${dataset} [Scenario: ${scen}]"
      echo "----------------------------------------------------"

      if [ "${dataset}" == "femnist" ]; then
        early_stop="${early_stop_femnist}"
      else
        early_stop="${early_stop_shakespeare}"
      fi

      for algorithm in "${algorithms[@]}"; do
        echo "Running simulations for algorithm: ${algorithm}..."

        if [ "${algorithm}" == "minibatch" ]; then
          for minibatch_val in "${minibatch_vals[@]}"; do
            for bg_model in "${bg_models[@]}"; do
              for tx_rate in "${transmission_success_rate[@]}"; do
                
                trace_file="${output_dir}/${scen}/${flops}/sys_metrics_${dataset}_${algorithm}_c_20_mb_${minibatch_val}.csv"
                trace_file_abs="$(realpath "${trace_file}")"

                echo "Launching parallel seeds for MB: ${minibatch_val} | BG: ${bg_model} | Tx: ${tx_rate}..."

                # Execução paralela apenas nas seeds
                for seed in "${seeds[@]}"; do
                  bg_model_lower=$(echo "${bg_model}" | tr '[:upper:]' '[:lower:]')
                  target_metrics="metrics_network_${scen}_${dataset}_minibatch_c_20_mb_${minibatch_val}_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"
                  stdout_log="trace_driven_${scen}_${dataset}_${algorithm}_c_20_mb_${minibatch_val}_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"

                  if [ -f "${PROJECT_ROOT}/${target_metrics}" ]; then
                    echo "=> Skipping (Already exists): [MB: ${minibatch_val} | Seed: ${seed}]"
                    continue
                  fi

                  wait_for_slot

                  (
                    # Diretório isolado por seed para evitar race condition na escrita do CSV
                    worker_dir=$(mktemp -d -p "${TMP_BASE}" "mb_${dataset}_${seed}_XXXXXX")
                    cd "${worker_dir}"

                    "${SIM_BIN_ABS}" -t "${trace_file_abs}" \
                                     -clients-b "${clients_bwd}" \
                                     -server-b "${server_bwd}" \
                                     -bg-workload "${bg_workload}" \
                                     -bg-model "${bg_model}" \
                                     -early-stop "${early_stop}" \
                                     -retransmission \
                                     -transmission-success-rate "${tx_rate}" \
                                     -seed "${seed}" \
                                     > "${PROJECT_ROOT}/${stdout_log}"

                    local_metric="metrics_network_${dataset}_minibatch_c_20_mb_${minibatch_val}.csv"
                    if [ -f "${local_metric}" ]; then
                      mv "${local_metric}" "${PROJECT_ROOT}/${target_metrics}"
                    fi

                    rm -rf "${worker_dir}"
                  ) &
                done
                wait # Aguarda todas as seeds deste bloco terminarem antes do próximo lote
              done
            done
          done

        else # Algoritmo FedAvg
          if [ "${dataset}" == "femnist" ]; then
            nclients_list=("${nclients_femnist[@]}")
          else
            nclients_list=("${nclients_shakespeare[@]}")
          fi

          for nclient in "${nclients_list[@]}"; do
            for bg_model in "${bg_models[@]}"; do
              for tx_rate in "${transmission_success_rate[@]}"; do

                trace_file="${output_dir}/${scen}/${flops}/sys_metrics_${dataset}_${algorithm}_c_${nclient}_e_1.csv"
                trace_file_abs="$(realpath "${trace_file}")"

                echo "Launching parallel seeds for Clients: ${nclient} | BG: ${bg_model} | Tx: ${tx_rate}..."

                # Execução paralela apenas nas seeds
                for seed in "${seeds[@]}"; do
                  bg_model_lower=$(echo "${bg_model}" | tr '[:upper:]' '[:lower:]')
                  target_metrics="metrics_network_${scen}_${dataset}_fedavg_c_${nclient}_e_1_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"
                  stdout_log="trace_driven_${scen}_${dataset}_${algorithm}_c_${nclient}_e_1_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"

                  if [ -f "${PROJECT_ROOT}/${target_metrics}" ]; then
                    echo "=> Skipping (Already exists): [Clients: ${nclient} | Seed: ${seed}]"
                    continue
                  fi

                  wait_for_slot

                  (
                    # Diretório isolado por seed
                    worker_dir=$(mktemp -d -p "${TMP_BASE}" "fedavg_${dataset}_${seed}_XXXXXX")
                    cd "${worker_dir}"

                    "${SIM_BIN_ABS}" -t "${trace_file_abs}" \
                                     -clients-b "${clients_bwd}" \
                                     -server-b "${server_bwd}" \
                                     -bg-workload "${bg_workload}" \
                                     -bg-model "${bg_model}" \
                                     -early-stop "${early_stop}" \
                                     -retransmission \
                                     -transmission-success-rate "${tx_rate}" \
                                     -seed "${seed}" \
                                     > "${PROJECT_ROOT}/${stdout_log}"

                    local_metric="metrics_network_${dataset}_fedavg_c_${nclient}_e_1.csv"
                    if [ -f "${local_metric}" ]; then
                      mv "${local_metric}" "${PROJECT_ROOT}/${target_metrics}"
                    fi

                    rm -rf "${worker_dir}"
                  ) &
                done
                wait # Aguarda todas as seeds deste bloco terminarem
              done
            done
          done
        fi
      done
    done
  done
done

rm -rf "${TMP_BASE}"
echo "Simulation script completed."