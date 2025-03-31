#!/bin/bash

# Define the lists
datasets=("femnist" "shakespeare")
algorithms=("fedavg" "minibatch")
nclients=(5 10 30 50)
minibatch_vals=(0.1 0.2 0.5 0.8)
flops_vals=(250000000 500000000 1000000000 1500000000 2000000000)

# Constants
clients_bwd=1500000000 # 1.5 Gbps
server_bwd=2250000000 # 2.25 Gbps
number_cores=4 # Paralelism level
p=0.95 # Portion of the program that can be parallelized

# Preprocess vars
output_dir="trace_driven_simulator/data/"

# Trap for SIGINT (Ctrl+C) and SIGTERM (kill) signals
cleanup() {
    echo "Caught interrupt signal. Cleaning up..."
    rm -rf "${output_dir}/homogeneus"
    rm -rf "${output_dir}/heterogeneus"
    exit 1
}

# Amdahl's Law function
amdahl_speedup() {
  local cores=$1
  local speedup=$(echo "scale=4; 1 / ((1 - $p) + ($p / $cores))" | bc -l)
  echo "${speedup}"
}

trap cleanup SIGINT SIGTERM

# Start logging
echo "Starting simulation script..."

# Homogeneous scenario
for flops in "${flops_vals[@]}"; do
  echo "Processing with FLOPs: ${flops}..."

  # Preprocessing data
  for dataset in "${datasets[@]}"; do
    echo "Processing dataset: ${dataset}..."

    if [ "$algorithm" == "shakespeare" ]; then
      echo "Running data processor for dataset Shakespeare with ${flops} FLOPs..."
      python3 trace_driven_simulator/data_processor.py --sample-dir "leaf_output/${dataset}/sys/" --search-pattern "metrics_sys_*" --output-dir "${output_dir}/homogeneus/${flops}/" --clients-flops "${flops}"
    else
      speedup=$(amdahl_speedup "${number_cores}")
      flops_adjusted=$(echo "${flops} * ${speedup}" | bc -l)
      flops_adjusted=${flops_adjusted%.*} # Convert to integer
      echo "Adjusted FLOPs with ${number_cores} cores: ${flops_adjusted}"
      echo "Running data processor for dataset FEMNIST with ${flops_adjusted} FLOPs..."
      python3 trace_driven_simulator/data_processor.py --sample-dir "leaf_output/${dataset}/sys/" --search-pattern "metrics_sys_*" --output-dir "${output_dir}/homogeneus/${flops}/" --clients-flops "${flops_adjusted}"
    fi
  done

  # Run simulations
  for dataset in "${datasets[@]}"; do
    echo "Starting simulations for dataset: ${dataset}..."

    for algorithm in "${algorithms[@]}"; do
      echo "Running simulation with algorithm: ${algorithm}..."

      if [ "$algorithm" == "minibatch" ]; then
        for minibatch_val in "${minibatch_vals[@]}"; do
          echo "Running Minibatch simulation with minibatch value: ${minibatch_val}..."

          trace_file="trace_driven_simulator/data/homogeneus/${flops}/metrics_sys_${dataset}_${algorithm}_c_30_mb_${minibatch_val}.csv"
          echo "Using trace file: ${trace_file}"

          go run trace_driven_simulator/main.go -t "${trace_file}" -clients-b "${clients_bwd}" -server-b "${server_bwd}" -bg-workload "${clients_bwd}" > "trace_driven_homogeneus_${dataset}_${algorithm}_c_30_mb_${minibatch_val}_fp_${flops}.csv"
          
          mv "metrics_network_${dataset}_minibatch_c_30_mb_${minibatch_val}.csv" "metrics_network_homogeneus_${dataset}_minibatch_c_30_mb_${minibatch_val}_fp_${flops}.csv"
          echo "Minibatch simulation complete. Results saved."
        done
      else
        for nclient in "${nclients[@]}"; do
          echo "Running FedAvg simulation with ${nclient} clients..."

          trace_file="trace_driven_simulator/data/homogeneus/${flops}/metrics_sys_${dataset}_${algorithm}_c_${nclient}_e_1.csv"
          echo "Using trace file: ${trace_file}"

          go run trace_driven_simulator/main.go -t "${trace_file}" -clients-b "${clients_bwd}" -server-b "${server_bwd}" -bg-workload "${clients_bwd}" > "trace_driven_homogeneus_${dataset}_${algorithm}_c_${nclient}_e_1_fp_${flops}.csv"

          mv "metrics_network_${dataset}_fedavg_c_${nclient}_e_1.csv" "metrics_network_homogeneus_${dataset}_fedavg_c_${nclient}_e_1_fp_${flops}.csv"
          echo "FedAvg simulation with ${nclient} clients complete. Results saved."
        done
      fi
    done
  done
  echo "Finished processing with FLOPs: ${flops}"
done

echo "Simulation script completed."