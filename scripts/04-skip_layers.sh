# datasets="arguana nfcorpus scidocs scifact fiqa webis-touche2020 trec-covid climate-fever msmarco"

datasets="nfcorpus scidocs scifact webis-touche2020 trec-covid climate-fever msmarco"

for dset in $datasets
do
	for i in $(seq 1 20)
	do
		echo $dset: skip layers start - $i
		CUDA_VISIBLE_DEVICES=2,3 python maggi/01_nvembed-skip-layers-001.py --dataset $dset --dset_type beir --batch_size 16 --skip_layer_start $i --num_layers_to_skip 10
		echo
	done
done

