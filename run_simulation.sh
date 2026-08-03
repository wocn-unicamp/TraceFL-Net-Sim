#!/bin/bash

# Define the lists
datasets=("femnist" "shakespeare")
algorithms=("fedavg" "minibatch")
nclients_femnist=(3 5 10 20 30 50)
nclients_shakespeare=(2 3 4 5 8 10 20)
minibatch_vals=(0.2 0.4 0.5 0.6 0.8 0.9 1)
flop_val=(250000000)

# Constants
clients_bwd=150000000 # 1.5 Gbps
server_bwd=2250000000 # 2.25 Gbps
bg_workload=0.67
number_cores=4 # Parallelism level
p_femnist=0.95 # Portion of the program that can be parallelized
p_shakespeare=0.4 # Portion of the program that can be parallelized

# Preprocess vars
output_dir="trace_driven_simulator/data"
sim_runner="go run trace_driven_simulator/main.go"

# Trap for SIGINT (Ctrl+C) and SIGTERM (kill) signals
cleanup() {
    echo "Caught interrupt signal. Cleaning up..."
    rm -rf "${output_dir}/homogeneus"
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

# Homogeneous scenario
for flops in "${flop_val[@]}"; do
  echo "Processing with FLOPs: ${flops}..."

  # --- Data Preprocessing ---
  for dataset in "${datasets[@]}"; do
    echo "Preprocessing data for dataset: ${dataset}..."
    
    # Removed 'local' scope keyword since we are in the main execution block
    if [ "${dataset}" == "shakespeare" ]; then
      p="${p_shakespeare}"
    else
      p="${p_femnist}"
    fi
    
    speedup=$(amdahl_speedup "${number_cores}" "${p}")
    
    # Perform multiplication and convert safely to integer using printf
    flops_adjusted=$(echo "${flops} * ${speedup}" | bc -l)
    flops_adjusted=$(printf "%.0f" "${flops_adjusted}")
    
    echo "Adjusted FLOPs with ${number_cores} cores: ${flops_adjusted}"
    echo "Running data processor for dataset ${dataset} with ${flops_adjusted} FLOPs..."
    python3 trace_driven_simulator/data_processor.py \
      --sample-dir "traces/sys" \
      --search-pattern "sys_metrics_*" \
      --output-dir "${output_dir}/homogeneus/${flops}/" \
      --clients-flops "${flops_adjusted}"
  done

  # --- Run simulations ---
  for dataset in "${datasets[@]}"; do
    echo "----------------------------------------------------"
    echo "Starting simulations for dataset: ${dataset}"
    echo "----------------------------------------------------"

    for algorithm in "${algorithms[@]}"; do
      echo "Running simulation with algorithm: ${algorithm}..."

      if [ "${algorithm}" == "minibatch" ]; then
        if [ "${dataset}" == "femnist" ]; then
            for minibatch_val in "${minibatch_vals[@]}"; do
              echo "Running Minibatch simulation with minibatch value: ${minibatch_val}..."

              trace_file="trace_driven_simulator/data/homogeneus/${flops}/sys_metrics_${dataset}_${algorithm}_c_20_mb_${minibatch_val}.csv"
              echo "Using trace file: ${trace_file}"

              ${sim_runner} -t "${trace_file}" -clients-b "${clients_bwd}" -server-b "${server_bwd}" -bg-workload "${bg_workload}" > "trace_driven_homogeneus_${dataset}_${algorithm}_c_20_mb_${minibatch_val}_fp_${flops}.csv"
              
              mv "metrics_network_${dataset}_minibatch_c_20_mb_${minibatch_val}.csv" "metrics_network_homogeneus_${dataset}_minibatch_c_20_mb_${minibatch_val}_fp_${flops}.csv"
              echo "Minibatch simulation complete. Results saved."
            done
        else # shakespeare
            for minibatch_val in "${minibatch_vals[@]}"; do
              echo "Running Minibatch simulation with minibatch value: ${minibatch_val}..."

              trace_file="trace_driven_simulator/data/homogeneus/${flops}/sys_metrics_${dataset}_${algorithm}_c_10_mb_${minibatch_val}.csv"
              echo "Using trace file: ${trace_file}"

              ${sim_runner} -t "${trace_file}" -clients-b "${clients_bwd}" -server-b "${server_bwd}" -bg-workload "${bg_workload}" > "trace_driven_homogeneus_${dataset}_${algorithm}_c_10_mb_${minibatch_val}_fp_${flops}.csv"
              
              mv "metrics_network_${dataset}_minibatch_c_10_mb_${minibatch_val}.csv" "metrics_network_homogeneus_${dataset}_minibatch_c_10_mb_${minibatch_val}_fp_${flops}.csv"
              echo "Minibatch simulation complete. Results saved."
            done
        fi
      else # This block handles FedAvg
        # Select the correct list of clients based on the dataset
        declare -a nclients_list
        if [ "${dataset}" == "femnist" ]; then
            nclients_list=("${nclients_femnist[@]}")
        else # shakespeare
            nclients_list=("${nclients_shakespeare[@]}")
        fi

        for nclient in "${nclients_list[@]}"; do
          echo "Running FedAvg simulation with ${nclient} clients..."

          trace_file="trace_driven_simulator/data/homogeneus/${flops}/sys_metrics_${dataset}_${algorithm}_c_${nclient}_e_1.csv"
          echo "Using trace file: ${trace_file}"

          ${sim_runner} -t "${trace_file}" -clients-b "${clients_bwd}" -server-b "${server_bwd}" -bg-workload "${bg_workload}" > "trace_driven_homogeneus_${dataset}_${algorithm}_c_${nclient}_e_1_fp_${flops}.csv"

          mv "metrics_network_${dataset}_fedavg_c_${nclient}_e_1.csv" "metrics_network_homogeneus_${dataset}_fedavg_c_${nclient}_e_1_fp_${flops}.csv"
          echo "FedAvg simulation with ${nclient} clients complete. Results saved."
        done
      fi
    done
  done
  echo "Finished processing with FLOPs: ${flops}"
done

echo "Simulation script completed."