# datasets="arguana fiqa msmarco nfcorpus scidocs scifact trec-covid webis-touche2020"
datasets="fiqa msmarco nfcorpus scidocs scifact trec-covid webis-touche2020"
dset_type="beir"

# datasets="musique"
# dset_type="multihop"

expt_no=3

if [ $expt_no == 1 ]
then
	for dataset in $datasets
	do
		echo $dataset
	
		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type
		# bash scripts/00-nvembed_inference.sh $dataset lbl $dset_type
	
		python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type
	done

elif [ $expt_no == 2 ]
then
	for dataset in $datasets
	do
		echo $dataset
	
		## gpt-category-linker

            	# qry_info_file=/data/datasets/$dset_type/metadata/$dataset/raw_data/test_gpt-category-linker.raw.csv
		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type category-gpt-linker None $qry_info_file 
		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --repr_suffix category-gpt-linker --save_suffix category-gpt-linker

		## category-gpt5-linker

		instruction="/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions_category-gpt5-linker.json"
            	qry_info_file="/data/datasets/$dset_type/metadata/$dataset/raw_data/test_category-gpt5-linker.csv"

		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type category-gpt5-linker $instruction $qry_info_file 
		python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --repr_suffix category-gpt5-linker --save_suffix category-gpt5-linker

	done

elif [ $expt_no == 3 ]
then
	for dataset in $datasets
	do
		echo $dataset
	
		instruction="/home/sasokan/suchith/maggi/instructions/01-beir_facts.json"

		bash scripts/00-nvembed_inference.sh $dataset fct $dset_type
		bash scripts/00-nvembed_inference.sh $dataset tst $dset_type  fact-lbl $instruction
		python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --fct_pred
	done
fi
