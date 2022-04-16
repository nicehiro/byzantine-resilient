#!/bin/bash

attacks=("max" "gaussian" "hidden" "litter" "empire")

for attack in ${attacks[@]}
do
    bash batch_train.sh -d MNIST -b 0.5 -c 0.4 -a $attack

    if [ $? -ne 0 ]
    then
        echo "Exit"
        exit 1
    fi
    echo $(date +"%m-%d %H:%M") ".........$connection_ratio/$byzantine_ratio: $attack-$par Done.........."
done
