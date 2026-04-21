# datasets="trec-covid climate-fever quora hotpotqa nq dbpedia-entity fever msmarco"
# dset_type="beir"

datasets="musique"
dset_type="multihop"

for dataset in $datasets
do
	echo $dataset

	bash scripts/00-nvembed_inference.sh $dataset tst $dset_type
	bash scripts/00-nvembed_inference.sh $dataset lbl $dset_type

	python maggi/00_nvembed-metric-from-embeddings-002.py --dataset $dataset --normalize --dset_type $dset_type
done

