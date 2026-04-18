import os, torch, json, torch.multiprocessing as mp, joblib, numpy as np, scipy.sparse as sp, argparse, math, torch.nn.functional as F, re
from typing import List

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


def compute_negatives(data_repr:sp.csr_matrix, data_mat:sp.csr_matrix, lbl_repr:sp.csr_matrix, data_pos:sp.csr_matrix, 
                      thresholds:List, data_batch_sz:int=1000, lbl_batch_sz:int=10_000):
    max_pos_scores = torch.tensor(data_pos.max(axis=1).toarray())

    score_info = {"data": {}, "indices": {}, "indptr": {}}
    for t in thresholds:
        for k,v in score_info.items(): v[t] = []

    for i in tqdm(range(0, data_repr.shape[0], data_batch_sz)):
        rep, gt, pos = data_repr[i:i+data_batch_sz].to('cuda'), data_mat[i:i+data_batch_sz], max_pos_scores[i:i+data_batch_sz]

        # Compute scores
        scores = []
        for j in tqdm(range(0, lbl_repr.shape[1], lbl_batch_sz)):
            sc = rep@lbl_repr[:, j:j+lbl_batch_sz].to('cuda')
            scores.append(sc.to('cpu'))
        scores = torch.hstack(scores)

        rows, cols = gt.nonzero()
        scores[rows, cols] = float("-inf")

        for t in thresholds:
            mask = scores >= pos * t
            scores[mask] = float("-inf")

            # Top-k scores
            sc, indices = torch.topk(scores, k=200, dim=1, largest=True)

            pred_score = sc.flatten().to(torch.float32)
            pred_idx = indices.flatten().to(torch.float32)
            pred_ptr = torch.full((len(rep),), 200, dtype=torch.int64)

            score_info["data"][t].append(pred_score)
            score_info["indices"][t].append(pred_idx)
            score_info["indptr"][t].append(pred_ptr)

    predictions = []
    for t in thresholds:
        all_data = torch.hstack(score_info["data"][t])
        all_indices = torch.hstack(score_info["indices"][t])
        all_indptr = torch.hstack([torch.zeros(1, dtype=torch.long), torch.hstack(score_info["indptr"][t]).cumsum(dim=0)])
        pred_lbl = sp.csr_matrix((all_data, all_indices, all_indptr), shape=(len(all_indptr)-1, lbl_repr.shape[1]), dtype=np.float32)
        predictions.append(pred_lbl)

    return predictions 


def additional_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--normalize', action='store_true')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    extra_args = additional_args()

    extra_args.dataset = "msmarco"
    extra_args.normalize = True

    # output_dir = "/home/sasokan/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    # output_dir = "/home/sasokan/b-sprabhu/outputs/mogicX/54_nvembed-for-msmarco-001/"
    output_dir = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    repr_dir = f"{output_dir}/representations/{extra_args.dataset}"
    lbl_repr = combine_embeddings(f"{repr_dir}/lbl_repr.pth", "lbl")

    trn_repr = combine_embeddings(f"{repr_dir}/trn_repr.pth", "trn")
    trn_repr = F.normalize(trn_repr, dim=1) if extra_args.normalize else trn_repr
    trn_lbl = sp.load_npz(f"/data/datasets/beir/{extra_args.dataset}/XC/trn_X_Y.npz")

    trn_pos = sp.load_npz("/home/sasokan/b-sprabhu/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/trn_X_Y_normalize-exact.npz")

    lbl_repr = F.normalize(lbl_repr, dim=1) if extra_args.normalize else lbl_repr
    lbl_repr = lbl_repr.T

    pred_dir = f"{output_dir}/predictions/{extra_args.dataset}"
    os.makedirs(pred_dir, exist_ok=True)

    # Threshold values in descending order.
    threshs = [0.95, 0.9, 0.7, 0.5]
    trn_negs = compute_negatives(trn_repr, trn_lbl, lbl_repr, trn_pos, threshs)

    for t, neg in tqdm(zip(threshs, trn_negs), total=len(threshs)):
        sp.save_npz(f"{pred_dir}/negatives_trn_X_Y_thresh-{int(t*100)}.npz", neg)

