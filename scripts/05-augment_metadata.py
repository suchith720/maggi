import scipy.sparse as sp, numpy as np, os, json

from sugar.core import load_raw_file, save_raw_file

from xclib.utils.sparse import retain_topk

if __name__ == "__main__":

    data_file = "/data/datasets/multihop/musique/XC/raw_data/test.raw.csv"
    data_ids, data_txt = load_raw_file(data_file)

    meta_file = "/data/datasets/multihop/musique/XC/raw_data/phrase.raw.csv"
    meta_ids, meta_txt = load_raw_file(meta_file)

    dm_file = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique/test_predictions_phrase.npz"
    data_meta = retain_topk(sp.load_npz(dm_file), k=5)

    aug_txt = []
    for q,r in zip(data_txt, data_meta):
        sort_idx = np.argsort(r.data)[::-1]
        indices = r.indices[sort_idx]
        txt = q + " [SEP] " + " [SEP] ".join([meta_txt[i] for i in indices])
        aug_txt.append(txt)

    save_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/raw_data/multihop/musique/"
    os.makedirs(save_dir, exist_ok=True)
    save_raw_file(f"{save_dir}/test_phrase_topk_sorted.raw.txt", data_ids, aug_txt)

    # Save examples

    np.random.seed(100)
    rnd_idx = np.random.permutation(len(data_txt))[:10]

    examples = []
    for idx in rnd_idx:
        sort_idx = np.argsort(data_meta[idx].data)[::-1]
        indices, scores = data_meta[idx].indices[sort_idx], data_meta[idx].data[sort_idx]
        example = {
            "query": data_txt[idx], 
            "phrases": [(meta_txt[i], float(s)) for i,s in zip(indices, scores)],
        }
        examples.append(example)

    fname = f"{save_dir}/examples_phrase_topk_sorted.json"
    with open(fname, "w") as file:
        json.dump(examples, file, indent=4)

