#!/bin/bash

if [ $# -lt 2 ]
then
	echo "scripts/00-nvembed_inference.sh <dataset> <role>"
	exit 1
fi

dataset=$1
role=$2

n_gpu=3
batch_size=16

for i in $(seq 0 $((n_gpu -1)))
do
	if [ $role == "lbl" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-to-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_lbl_repr --dataset $dataset --batch_size $batch_size &

	elif [ $role == "tst" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-to-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_tst_repr --dataset $dataset --batch_size $batch_size &

	elif [ $role == "trn" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-to-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_trn_repr --dataset $dataset --batch_size $batch_size &

	else
		echo "Invalid role: $role"
	fi
done

wait
