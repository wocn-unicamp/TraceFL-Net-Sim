#!/usr/bin/env bash

output_dir="${1:-./baseline}"

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="1000"

fedavg_lr="0.004"
declare -a fedavg_vals=( "5 1" "10 1" "30 1" "50 1")

minibatch_lr="0.06"
declare -a minibatch_vals=( "30 0.1"  "30 0.2"  "30 0.5"  "30 0.8")

###################### Functions ###################################

function move_data() {
    path="$1"
    suffix="$2"
    
    pushd models/metrics
        mv metrics_sys.csv "${path}/metrics_sys_${suffix}.csv"
		mv metrics_stat.csv "${path}/metrics_stat_${suffix}.csv"
    popd

    cp -rf data/shakespeare/meta "${path}"
    mv "${path}/meta" "${path}/meta_${suffix}"
}

function run_fedavg() {
	clients_per_round="$1"
	num_epochs="$2"

	pushd models/
		python3.6 main.py -dataset 'shakespeare' -model 'stacked_lstm' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
	popd
	move_data ${output_dir} "shakespeare_fedavg_c_${clients_per_round}_e_${num_epochs}"
}

function run_minibatch() {
	clients_per_round="$1"
	minibatch_percentage="$2"

	pushd models/
		python3.6 main.py -dataset 'shakespeare' -model 'stacked_lstm' --minibatch ${minibatch_percentage} --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} -lr ${minibatch_lr}
	popd
	move_data ${output_dir} "shakespeare_minibatch_c_${clients_per_round}_mb_${minibatch_percentage}"
}

##################### Script #################################

pushd ../

# Check that data and models are available
if [ ! -d 'data/' -o ! -d 'models/' ]; then
    echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# Check that output directory is available
mkdir -p ${output_dir}
output_dir=`realpath ${output_dir}`
echo "Storing results in directory ${output_dir} (please invoke this script as: ${0} <dirname> to change)"

# If data unavailable, execute pre-processing script
if [ ! -d 'data/shakespeare/data/train' ]; then
    if [ ! -f 'data/shakespeare/preprocess.sh' ]; then
        echo "Couldn't find data/ and/or models/ directories " \
             "- please obtain scripts from GitHub repo: https://github.com/TalwalkarLab/leaf"
        exit 1
    fi

    echo "Couldn't find Shakespeare data - " \
         "running data preprocessing script"
    pushd data/shakespeare/
        rm -rf meta/ data/all_data data/test data/train data/rem_user_data data/intermediate
        ./preprocess.sh -s niid --sf 0.05 -k 64 -tf 0.9 -t sample
    popd
fi

# Run minibatch SGD experiments
for val_pair in "${minibatch_vals[@]}"; do
	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
	minibatch_percentage=`echo ${val_pair} | cut -d' ' -f2`
	echo "Running Minibatch experiment with fraction ${minibatch_percentage} and ${clients_per_round} clients"
	run_minibatch "${clients_per_round}" "${minibatch_percentage}"
done

# Run FedAvg experiments
for val_pair in "${fedavg_vals[@]}"; do
	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
	num_epochs=`echo ${val_pair} | cut -d' ' -f2`
	echo "Running FedAvg experiment with ${num_epochs} local epochs and ${clients_per_round} clients"
	run_fedavg "${clients_per_round}" "${num_epochs}"
done

popd
