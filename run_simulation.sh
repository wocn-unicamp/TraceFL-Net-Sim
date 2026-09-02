#!/bin/bash

# Define the lists
datasets=("femnist" "shakespeare")
algorithms=("fedavg" "minibatch")
# Options: POISSON PARETO CBR MULTI
bg_models=("MULTI")
trasmission_sucess_rate=(1.0)
# Options homogeneous heterogeneous
scenarios=("heterogeneous")
nclients_femnist=(3 5 10 20 30 50)
nclients_shakespeare=(2 3 4 5 8 10 20)
minibatch_vals=(0.2 0.4 0.5 0.6 0.8 0.9 1)
flop_val=(500000000)
seeds=(42 1337 2026 8888 9999) # Five distinct seeds for statistical reliability

# Constants
clients_bwd=1500000000 # 1.5 Gbps
server_bwd=2250000000 # 2.25 Gbps
bg_workload=0.67
number_cores=4 # Parallelism level
p_femnist=0.95 # Portion of the program that can be parallelized
p_shakespeare=0.4 # Portion of the program that can be parallelized
early_stop_femnist=278
early_stop_shakespeare=-1

# Preprocess vars
output_dir="trace_driven_simulator/data"
sim_runner="go run trace_driven_simulator/main.go"

# Trap for SIGINT (Ctrl+C) and SIGTERM (kill) signals
cleanup() {
    echo "Caught interrupt signal. Cleaning up..."
    rm -rf "${output_dir}"
    exit 1
}

# Amdahl's Law function
amdahl_speedup() {
  local cores=$1
  local p=$2
  echo "scale=4; 1 / ((1 - $p) + ($p / $cores))" | bc -l
}

trap cleanup SIGINT SIGTERM

# Start logging
echo "Starting simulation script..."

for flops in "${flop_val[@]}"; do
  echo "Processing with FLOPs: ${flops}..."

  for scen in "${scenarios[@]}"; do
    echo "===================================================="
    echo "Processing Scenario: ${scen}"
    echo "===================================================="

    flops_mode="${scen}"

    # --- Data Preprocessing ---
    for dataset in "${datasets[@]}"; do
      echo "Preprocessing data for dataset: ${dataset} [Scenario: ${scen}]..."
      
      if [ "${dataset}" == "shakespeare" ]; then
        p="${p_shakespeare}"
      else
        p="${p_femnist}"
      fi
      
      speedup=$(amdahl_speedup "${number_cores}" "${p}")
      
      # Perform multiplication and convert safely to integer using printf
      flops_adjusted=$(echo "${flops} * ${speedup}" | bc -l)
      flops_adjusted=$(printf "%.0f" "${flops_adjusted}")

      offset_adjusted=$(echo "300000000 * ${speedup}" | bc -l)
      offset_adjusted=$(printf "%.0f" "${offset_adjusted}")

      std1_adjusted=$(echo "50000000 * ${speedup}" | bc -l)
      std1_adjusted=$(printf "%.0f" "${std1_adjusted}")

      std2_adjusted=$(echo "100000000 * ${speedup}" | bc -l)
      std2_adjusted=$(printf "%.0f" "${std2_adjusted}")
      
      echo "Adjusted FLOPs with ${number_cores} cores: ${flops_adjusted}"
      echo "Running data processor for dataset ${dataset} with ${flops_adjusted} FLOPs (Mode: ${flops_mode})..."
      
      # Python invocation passing --flops-mode
      python3 trace_driven_simulator/data_processor.py \
        --sample-dir "traces/sys" \
        --search-pattern "sys_metrics_${dataset}_*" \
        --output-dir "${output_dir}/${scen}/${flops}/" \
        --flops-mode "${flops_mode}" \
        --clients-flops "${flops_adjusted}"\
        --bimodal-offset "${offset_adjusted}" \
        --bimodal-std1 "${std1_adjusted}" \
        --bimodal-std2 "${std2_adjusted}"
    done

    # --- Run simulations ---
    for dataset in "${datasets[@]}"; do
      echo "----------------------------------------------------"
      echo "Starting simulations for dataset: ${dataset} [Scenario: ${scen}]"
      echo "----------------------------------------------------"

      for algorithm in "${algorithms[@]}"; do
        echo "Running simulation with algorithm: ${algorithm}..."

        if [ "${dataset}" == "femnist" ]; then
            early_stop="${early_stop_femnist}"
        else # shakespeare
            early_stop="${early_stop_shakespeare}"
        fi

        if [ "${algorithm}" == "minibatch" ]; then
          for minibatch_val in "${minibatch_vals[@]}"; do
            for bg_model in "${bg_models[@]}"; do
              for tx_rate in "${trasmission_sucess_rate[@]}"; do
                for seed in "${seeds[@]}"; do
                  bg_model_lower=$(echo "${bg_model}" | tr '[:upper:]' '[:lower:]')
                  
                  target_metrics="metrics_network_${scen}_${dataset}_minibatch_c_20_mb_${minibatch_val}_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"
                  
                  if [ -f "${target_metrics}" ]; then
                    echo "=> Skipping experiment (Already exists): Dataset: ${dataset} | MB: ${minibatch_val} | BG Model: ${bg_model} | Tx Rate: ${tx_rate} | Seed: ${seed}"
                    continue
                  fi

                  echo "Running Minibatch simulation | Scenario: ${scen} | Dataset: ${dataset} | MB: ${minibatch_val} | BG Model: ${bg_model} | Tx Rate: ${tx_rate} | Seed: ${seed}..."

                  trace_file="${output_dir}/${scen}/${flops}/sys_metrics_${dataset}_${algorithm}_c_20_mb_${minibatch_val}.csv"
                  echo "Using trace file: ${trace_file}"

                  ${sim_runner} -t "${trace_file}" \
                                -clients-b "${clients_bwd}" \
                                -server-b "${server_bwd}" \
                                -bg-workload "${bg_workload}" \
                                -bg-model "${bg_model}" \
                                -early-stop "${early_stop}" \
                                -retransmission \
                                -transmission-success-rate "${tx_rate}" \
                                -seed "${seed}" \
                                > "trace_driven_${scen}_${dataset}_${algorithm}_c_20_mb_${minibatch_val}_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"
                  
                  mv "metrics_network_${dataset}_minibatch_c_20_mb_${minibatch_val}.csv" \
                     "${target_metrics}"
                  
                  echo "Minibatch simulation complete for BG Model ${bg_model}, Tx Rate ${tx_rate}, Seed ${seed}. Results saved."
                done
              done
            done
          done
        else # This block handles FedAvg
          # Select the correct list of clients based on the dataset
          declare -a nclients_list
          if [ "${dataset}" == "femnist" ]; then
              nclients_list=("${nclients_femnist[@]}")
          else # shakespeare
              nclients_list=("${nclients_shakespeare[@]}")
          fi

          for nclient in "${nclients_list[@]}"; do
            for bg_model in "${bg_models[@]}"; do
              for tx_rate in "${trasmission_sucess_rate[@]}"; do
                for seed in "${seeds[@]}"; do
                  bg_model_lower=$(echo "${bg_model}" | tr '[:upper:]' '[:lower:]')
                  
                  target_metrics="metrics_network_${scen}_${dataset}_fedavg_c_${nclient}_e_1_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"

                  if [ -f "${target_metrics}" ]; then
                    echo "=> Skipping experiment (Already exists): Dataset: ${dataset} | Clients: ${nclient} | BG Model: ${bg_model} | Tx Rate: ${tx_rate} | Seed: ${seed}"
                    continue
                  fi

                  echo "Running FedAvg simulation | Scenario: ${scen} | Dataset: ${dataset} | Clients: ${nclient} | BG Model: ${bg_model} | Tx Rate: ${tx_rate} | Seed: ${seed}..."

                  trace_file="${output_dir}/${scen}/${flops}/sys_metrics_${dataset}_${algorithm}_c_${nclient}_e_1.csv"
                  echo "Using trace file: ${trace_file}"

                  ${sim_runner} -t "${trace_file}" \
                                -clients-b "${clients_bwd}" \
                                -server-b "${server_bwd}" \
                                -bg-workload "${bg_workload}" \
                                -bg-model "${bg_model}" \
                                -early-stop "${early_stop}" \
                                -retransmission \
                                -transmission-success-rate "${tx_rate}" \
                                -seed "${seed}" \
                                > "trace_driven_${scen}_${dataset}_${algorithm}_c_${nclient}_e_1_bg_${bg_model_lower}_tx_${tx_rate}_fp_${flops}_seed_${seed}.csv"

                  mv "metrics_network_${dataset}_fedavg_c_${nclient}_e_1.csv" \
                     "${target_metrics}"
                  
                  echo "FedAvg simulation with ${nclient} clients, BG Model ${bg_model}, Tx Rate ${tx_rate}, Seed ${seed} complete. Results saved."
                done
              done
            done
          done
        fi
      done
    done
  done
  echo "Finished processing with FLOPs: ${flops}"
done

echo "Simulation script completed."