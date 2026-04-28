# datasets="arguana nfcorpus scidocs scifact fiqa webis-touche2020 trec-covid msmarco"
datasets="arguana"
expt_no=2

if [ $expt_no -eq 1 ]
then
	for dset in $datasets
	do
		for i in $(seq 1 20)
		do
			echo $dset: skip layers start - $i
			CUDA_VISIBLE_DEVICES=2,3 python maggi/01_nvembed-skip-layers-001.py --dataset $dset --dset_type beir --batch_size 16 --skip_layer_start $i --num_layers_to_skip 10
			echo
		done
	done

elif [ $expt_no -eq 2 ]
then
	funcs="modulo-2 modulo-3 modulo-4 modulo-5 start_concentrated middle_concentrated end_concentrated"
	
	for dset in $datasets
	do
		for func in $funcs
		do
			echo $dset: skip function - $func 
			CUDA_VISIBLE_DEVICES=2,3 python maggi/01_nvembed-skip-layers-002.py --dataset $dset --dset_type beir --batch_size 16 --skip_func $func
			echo
		done
	done

else
	echo "Invalid experiment number."
fi
