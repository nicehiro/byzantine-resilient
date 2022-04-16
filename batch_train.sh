#!/bin/bash
# train a batch of experiments

# set parameters
# copied from https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
helpFunction()
{
   echo ""
   echo "Usage: $0 -d dataset -b byzantine_ratio -c connection_ratio -a attack"
   exit 1 # Exit script after printing help
}

while getopts "d:b:c:a:" opt
do
    echo "$opt"
    case "$opt" in
        d ) dataset="$OPTARG" ;;
        b ) byzantine_ratio="$OPTARG" ;;
        c ) connection_ratio="$OPTARG" ;;
        a ) attack="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

# Print helpFunction in case parameters are empty
if [ -z "$dataset" ] || [ -z "$byzantine_ratio" ] || [ -z "$connection_ratio" ] || [ -z "$attack" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


if [ $dataset = "MNIST" ]
then
    epochs=50
    basedir="mnist"
else
    epochs=200
    basedir="cifar"
fi

pars=("average" "bridge" "median" "krum" "bulyan" "zeno" "mozi" "qc")

for par in ${pars[@]}
do
    logdir="$basedir/$attack-$par"
    python train.py \
        --epochs $epochs \
        --dataset $dataset \
        --batch_size 256 \
        --nodes_n 30 \
        --byzantine_ratio $byzantine_ratio \
        --connection_ratio $connection_ratio \
        --attack $attack \
        --par $par \
        --logdir $logdir

    if [ $? -ne 0 ]
    then
        echo "Exit"
        exit 1
    fi
    echo $(date +"%m-%d %H:%M") ".........$connection_ratio/$byzantine_ratio: $attack-$par Done.........."
done

