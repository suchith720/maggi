datasets="scidocs fiqa nfcorpus webis-touche2020 hotpotqa nq quora trec-covid msmarco climate-fever dbpedia-entity fever"

for dataset in $datasets
do
	echo $dataset

	bash scripts/00-nvembed_inference.sh $dataset tst
	bash scripts/00-nvembed_inference.sh $dataset lbl

	python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize
done

