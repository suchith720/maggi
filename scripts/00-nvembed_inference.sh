#!/bin/bash

if [ $# -lt 3 ]
then
	echo "scripts/00-nvembed_inference.sh <dataset> <role> <dset_type> <optional:save_suffix> <optional:instruction> <optional:qry_info_file>"
	exit 1
fi

dataset=$1
role=$2
dset_type=$3

if [ $# -gt 3 ]
then
	save_suffix=$4
else
	save_suffix=None
fi

if [ $# -gt 4 ]
then
	instruction=$5
else
	instruction=None
fi

if [ $# -gt 5 ]
then
	qry_info_file=$6
else
	qry_info_file=None
fi

n_gpu=4
batch_size=16

for i in $(seq 0 $((n_gpu -1)))
do
	if [ $role == "lbl" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_lbl_repr --dataset $dataset --batch_size $batch_size --dset_type $dset_type &

	elif [ $role == "phr" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_phr_repr --dataset $dataset --batch_size $batch_size --dset_type $dset_type &

	elif [ $role == "tst" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_tst_repr --dataset $dataset --batch_size $batch_size --dset_type $dset_type \
			--save_suffix $save_suffix --instruction $instruction --qry_info_file $qry_info_file &

	elif [ $role == "trn" ]
	then
		CUDA_VISIBLE_DEVICES=$i python maggi/00_nvembed-compute-msmarco-embeddings-001.py --idx $i --parts $n_gpu --get_trn_repr --dataset $dataset --batch_size $batch_size --dset_type $dset_type &

	else
		echo "Invalid role: $role"
	fi
done

wait
