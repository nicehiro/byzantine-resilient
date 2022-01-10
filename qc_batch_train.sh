#!/bin/bash
# train a batch of qc experiments under different connection ratio and byzantine ratio

# set parameters
# copied from https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
helpFunction() {
    echo ""
    echo "Usage: $0"
    exit 1 # Exit script after printing help
}

dataset="MNIST"
epochs=50
basedir="mnist/qc"

attacks=("gaussian" "max" "hidden" "litter" "empire")
crs=(0.2 0.4 0.6)
brs=(0.1 0.3 0.5)
len_cr=${#crs[@]}
len_br=${#brs[@]}
len_att=${#attacks[@]}

for ((i = 0; i < $len_cr; i++)); do
    for ((j = 0; j < $len_br; j++)); do
        for ((k = 0; k < $len_att; k++)); do
            cr=${crs[i]}
            br=${brs[j]}
            attack=${attacks[k]}
            mkdir -p "logs/$basedir/$cr-$br/"
            logdir="$basedir/$cr-$br/$attack-qc"
            python train.py \
                --epochs $epochs \
                --dataset $dataset \
                --batch_size 128 \
                --nodes_n 30 \
                --byzantine_ratio $br \
                --connection_ratio $cr \
                --attack $attack \
                --par qc \
                --logdir $logdir

            if [ $? -ne 0 ]; then
                echo "Exit"
                exit 1
            fi
            echo $(date +"%m-%d %H:%M") ".......$connection_ratio/$byzantine_ratio: $attack-$par Done........"
        done
    done
done
