datasets="trec-covid climate-fever quora hotpotqa nq dbpedia-entity fever msmarco"

for dataset in $datasets
do
	echo $dataset

	bash scripts/00-nvembed_inference.sh $dataset tst
	bash scripts/00-nvembed_inference.sh $dataset lbl

	python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize
done

