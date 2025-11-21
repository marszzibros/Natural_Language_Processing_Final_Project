#!/bin/bash

model_ids=("Qwen/Qwen3-VL-30B-A3B-Instruct")
cities=("Chicago" "SanFrancisco" "NewYork" "Seattle") 
modes=("satellite" "panorama")

for model_id in "${model_ids[@]}"; do
    for city in "${cities[@]}"; do
        for mode in "${modes[@]}"; do
            sbatch inference.sh "$model_id" "$city" "$mode"
        done
    done
done
