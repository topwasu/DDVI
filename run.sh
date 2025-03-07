#!/bin/bash

dataset=$1
model=$2
prior=$3

mkdir experimental_results/
mkdir experimental_results/$dataset_$model_$prior
python run.py dataset=$dataset model=$model prior=$prior save_folder=${save_folder}/run0 seed=0