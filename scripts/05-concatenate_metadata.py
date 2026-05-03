import scipy.sparse as sp, numpy as np, os, json

from sugar.core import load_raw_file, save_raw_file

from xclib.utils.sparse import retain_topk

def musique_metadata():
    data_file = "/data/datasets/multihop/musique/XC/raw_data/test.raw.csv"
    data_ids, data_txt = load_raw_file(data_file)

    meta_file = "/data/datasets/multihop/musique/XC/raw_data/phrase.raw.csv"
    meta_ids, meta_txt = load_raw_file(meta_file)

    dm_file = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique/test_phrases.npz"
    data_meta = retain_topk(sp.load_npz(dm_file), k=5)

    meta_order = "random"

    # ----------------------------------------
    # ----------------------------------------

    aug_txt = []
    for q,r in zip(data_txt, data_meta):

        if meta_order == "sorted":
            idx = np.argsort(r.data)[::-1]
        elif meta_order == "random":
            idx = np.random.permutation(len(r.data))
        else:
            raise ValueError(f"Invalid order type: {meta_order}.")

        indices = r.indices[idx]
        txt = q + " [SEP] " + " [SEP] ".join([meta_txt[i] for i in indices])
        aug_txt.append(txt)

    save_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/raw_data/multihop/musique/"
    os.makedirs(save_dir, exist_ok=True)

    if meta_order == "sorted":
        raw_file = f"{save_dir}/test_phrase_topk-sorted.raw.txt"
        exp_file = f"{save_dir}/examples_phrase_topk-sorted.json"
    elif meta_order == "random":
        raw_file = f"{save_dir}/test_phrase_topk-random.raw.txt"
        exp_file = f"{save_dir}/examples_phrase_topk-random.json"

    save_raw_file(raw_file, data_ids, aug_txt)

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

    with open(exp_file, "w") as file:
        json.dump(examples, file, indent=4)


from typing import List, Optional
def save_file_for_generations(fname, inputs:List, sep:Optional[str]="\t", encoding:Optional[str]='utf-8'):
    sizes = np.array([len(o) for o in inputs])
    assert np.all(sizes == sizes[0]), "Number of elements in each input should be the same."
    with open(fname, 'w', encoding=encoding) as file:
        for o in zip(*inputs):
            line = ""
            for i in o:
                i = str(i).replace("\n", "").replace("\t", "").replace("->", "")
                line = line + sep + i if len(line) else i
            file.write(f'{line}\n')

def musique_metadata_filtering():
    data_file = "/data/datasets/multihop/musique/XC/raw_data/test.raw.csv"
    data_ids, data_txt = load_raw_file(data_file)

    meta_file = "/data/datasets/multihop/musique/XC/raw_data/phrase.raw.csv"
    meta_ids, meta_txt = load_raw_file(meta_file)

    dm_file = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/multihop/musique/test_phrases.npz"
    data_meta = retain_topk(sp.load_npz(dm_file), k=5)

    data_meta_txt = [json.dumps([meta_txt[i] for i in o.indices]) for o in data_meta]
    assert len(data_meta_txt) == len(data_txt)

    save_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/raw_data/multihop/musique/"
    save_file_for_generations(f"{save_dir}/test_phrase_filtering.raw.txt", [data_ids, data_txt, data_meta_txt])


if __name__ == "__main__":
    # musique_metadata()

    musique_metadata_filtering()

