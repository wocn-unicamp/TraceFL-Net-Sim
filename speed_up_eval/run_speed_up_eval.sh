#!/bin/bash

# -------------------------------------------------------------------
# Script Name: run_speed_up_eval.sh
# Description: Runs the Docker command with varying CPU cores and
#              stores the output in a text file.
# Usage:       ./run_speed_up_eval.sh
# -------------------------------------------------------------------

cores=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
output_file="speed_up_results.txt"
timestamp=$(date +"%Y-%m-%d %H:%M:%S")

# Initialize or clear the output file
cat <<EOF > "$output_file"
Speed-Up Evaluation Results
Timestamp: $timestamp
----------------------------------------
EOF

# Loop through each core value and execute the Docker command
for core in "${cores[@]}"; do
    echo "Running Docker container with ${core} cores..."
    docker_output=$(docker run --rm --cpus "${core}" speed_up_eval 2>&1)
    
    # Append output or error to the file
    {
        echo "Core: ${core}"
        echo "$([[ $? -eq 0 ]] && echo "Output:" || echo "Error:")"
        echo "$docker_output"
        echo "----------------------------------------"
    } >> "$output_file"
    
    echo "Completed for ${core} cores."
    sleep 1
done

echo "All Docker runs completed. Results saved to ${output_file}."