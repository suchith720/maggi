import os, torch, json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse, math, torch.nn.functional as F, re

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from xcai.metrics import *


def _get_num_parts(dirname:str, role:str):
    return max([int(re.match(role + r"_repr_([0-9]{3}).pth", o)[1]) for o in os.listdir(dirname) if re.match(role + r"_repr_([0-9]{3}).pth", o)]) + 1


def combine_embeddings(fname:str, role:str):
    if os.path.exists(fname):
        rep = torch.load(fname)
    else:
        output_dir = os.path.dirname(fname)
        num_parts = _get_num_parts(output_dir, role)
        rep = torch.vstack([torch.load(f'{output_dir}/{role}_repr_{idx:03d}.pth') for idx in range(num_parts)])
        torch.save(rep, fname)
    return rep


def compute_metrics(data_repr:sp.csr_matrix, data_mat:sp.csr_matrix, lbl_repr:sp.csr_matrix, data_batch_sz:int=1000, lbl_batch_sz:int=10_000):
    metric = PrecReclMrr(lbl_repr.shape[1], pk=10, rk=200, rep_pk=[1, 3, 5, 10], rep_rk=[10, 100, 200], mk=[5, 10, 20])
    metric.reset()

    all_data, all_indices, all_indptr = [], [], []

    for i in tqdm(range(0, data_repr.shape[0], data_batch_sz)):
        rep, gt = data_repr[i:i+data_batch_sz].to('cuda'), data_mat[i:i+data_batch_sz]

        # Compute scores
        scores = []
        for j in tqdm(range(0, lbl_repr.shape[1], lbl_batch_sz)):
            sc = rep@lbl_repr[:, j:j+lbl_batch_sz].to('cuda')
            scores.append(sc.to('cpu'))
        scores = torch.hstack(scores)

        # Top-k scores
        scores, indices = torch.topk(scores, k=200, dim=1, largest=True)
        o = {
            'pred_score': scores.flatten().to(torch.float32),
            'pred_idx': indices.flatten().to(torch.float32),
            'pred_ptr': torch.full((len(rep),), 200, dtype=torch.int64),
            'targ_idx': torch.tensor(gt.indices, dtype=torch.int64),
            'targ_score': torch.tensor(gt.data, dtype=torch.float32),
            'targ_ptr': torch.tensor([p-q for p,q in zip(gt.indptr[1:], gt.indptr)], dtype=torch.int64),
        }
        metric.accumulate(**o)

        all_data.append(o["pred_score"])
        all_indices.append(o["pred_idx"])
        all_indptr.append(o["pred_ptr"])

    all_data = torch.hstack(all_data)
    all_indices = torch.hstack(all_indices)
    all_indptr = torch.hstack([torch.zeros(1, dtype=torch.long), torch.hstack(all_indptr).cumsum(dim=0)])
    pred_lbl = sp.csr_matrix((all_data, all_indices, all_indptr), dtype=np.float32)

    return {k:float(v) for k,v in metric.value.items()}, pred_lbl


def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    extra_args = additional_args()

    # output_dir = "/home/sasokan/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    # output_dir = "/home/sasokan/b-sprabhu/outputs/mogicX/54_nvembed-for-msmarco-001/"

    output_dir = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    repr_dir = f"{output_dir}/representations/{extra_args.dataset}"
    lbl_repr = combine_embeddings(f"{repr_dir}/lbl_repr.pth", "lbl")

    tst_repr = combine_embeddings(f"{repr_dir}/tst_repr.pth", "tst")
    tst_repr = F.normalize(tst_repr, dim=1) if extra_args.normalize else tst_repr
    tst_lbl = sp.load_npz(f"/data/datasets/beir/{extra_args.dataset}/XC/tst_X_Y.npz")

    if extra_args.train:
        trn_repr = combine_embeddings(f"{repr_dir}/trn_repr.pth", "trn")
        trn_repr = F.normalize(trn_repr, dim=1) if extra_args.normalize else trn_repr
        trn_lbl = sp.load_npz(f"/data/datasets/beir/{extra_args.dataset}/XC/trn_X_Y.npz")

    lbl_repr = F.normalize(lbl_repr, dim=1) if extra_args.normalize else lbl_repr
    lbl_repr = lbl_repr.T

    pred_dir = f"{output_dir}/predictions/{extra_args.dataset}"
    os.makedirs(pred_dir, exist_ok=True)

    metrics, tst_pred = compute_metrics(tst_repr, tst_lbl, lbl_repr)
    sp.save_npz(f"{pred_dir}/test_predictions.npz", tst_pred)

    if extra_args.train:
        m, trn_pred = compute_metrics(trn_repr, trn_lbl, lbl_repr)
        metrics = {"train": m, "test": metrics}
        sp.save_npz(f"{pred_dir}/train_predictions.npz", trn_pred)

    os.makedirs(f"{output_dir}/metrics", exist_ok=True)
    metric_file = f"{output_dir}/metrics/{extra_args.dataset}.json"
    with open(metric_file, "w") as file:
        json.dump(metrics, file, indent=4)

    print(metrics)



