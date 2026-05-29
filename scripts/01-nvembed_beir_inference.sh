datasets="fiqa msmarco nfcorpus scidocs scifact trec-covid webis-touche2020"

datasets="msmarco"
dset_type="beir"
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
		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type \
		# --repr_suffix category-gpt-linker --save_suffix category-gpt-linker

		## category-gpt5-linker

		# instruction="/home/sasokan/suchith/xcai/xcai/models/nvembed/instructions_category-gpt5-linker.json"
            	# qry_info_file="/data/datasets/$dset_type/metadata/$dataset/raw_data/test_category-gpt5-linker.csv"

		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type category-gpt5-linker $instruction $qry_info_file 
		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type \
		# --repr_suffix category-gpt5-linker --save_suffix category-gpt5-linker

		## HippoRAG metadata
		
		# instruction="instructions/02-hipporag-fact-nvembedv2.json"
                # qry_info_file="/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-002/raw_data/$dset_type/$dataset/test_fact_topk-sorted.raw.txt"
                # bash scripts/00-nvembed_inference.sh $dataset tst $dset_type hipporag-fact-nvembedv2 $instruction $qry_info_file

		lbl_rep_file="/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/representations/$dset_type/$dataset/lbl_repr.pth"
                python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type \
			--repr_suffix hipporag-fact-nvembedv2 --save_suffix hipporag-fact-nvembedv2 --lbl_rep_file $lbl_rep_file

	done

elif [ $expt_no == 3 ]
then
	for dataset in $datasets
	do
		echo $dataset
	
		instruction="/home/sasokan/suchith/maggi/instructions/01-beir_facts.json"

		# bash scripts/00-nvembed_inference.sh $dataset fct $dset_type
		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type  fact-lbl $instruction
		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --fct_pred

		# bash scripts/00-nvembed_inference.sh $dataset int $dset_type
		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type  intent-lbl $instruction
		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --int_pred

		meta_info_file=/data/datasets/beir/$dataset/XC/raw_data/hipporag-fact_exact.raw.csv
		bash scripts/00-nvembed_inference.sh $dataset meta $dset_type None None None hipporag-fact-exact $meta_info_file

		# bash scripts/00-nvembed_inference.sh $dataset tst $dset_type  fact-lbl $instruction
		# bash scripts/00-nvembed_inference.sh $dataset trn $dset_type  fact-lbl $instruction

		# python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --int_pred --train
	done

elif [ $expt_no == 4 ]
then
	for dataset in $datasets
	do
		echo $dataset
	
		python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type --fct_pred --similarity
	done
fi

