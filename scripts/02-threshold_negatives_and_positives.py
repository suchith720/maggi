import numpy as np, scipy.sparse as sp, os
from tqdm.auto import tqdm
from sugar.core import *

from xclib.utils.sparse import retain_topk

def map_from_negatives_to_full():
    fname = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"
    lbl_ids, lbl_txt = load_raw_file(fname)
    lbl_txt2idx = {t.replace("->", ""):i for i,t in enumerate(lbl_txt)}

    fname = "/data/datasets/beir/msmarco/XC/raw_data/ce-scores.raw.txt"
    elbl_ids, elbl_txt = load_raw_file(fname)

    return [lbl_txt2idx[l] for l in elbl_txt]

def map_from_exact_to_full():
    fname = "/data/datasets/beir/msmarco/XC/raw_data/label.raw.csv"
    lbl_ids, lbl_txt = load_raw_file(fname)
    lbl_txt2idx = {t:i for i,t in enumerate(lbl_txt)}

    fname = "/data/datasets/beir/msmarco/XC/raw_data/label_exact.raw.txt"
    elbl_ids, elbl_txt = load_raw_file(fname)

    return [lbl_txt2idx[l] for l in elbl_txt]


if __name__ == "__main__":
    expt_no = 1

    # Experiments 0 to 2 focused more on NV-Embed-v2 negatives.
    # Experiments 3 onwards are about using Cross encoder scores, which should have been the obvious first step.

    if expt_no == 0:
        thresh = 0.5

        fname = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/trn_X_Y_normalize-exact.npz"
        pos_lbl = sp.load_npz(fname)

        neg_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize.npz"
        neg_lbl = sp.load_npz(neg_file)

        mask = np.hstack([n.data > p.max() * thresh for n,p in tqdm(zip(neg_lbl, pos_lbl), total=pos_lbl.shape[0])])
        neg_lbl.data[mask] = 0.0
        neg_lbl.eliminate_zeros()

        save_dir = os.path.dirname(neg_file)
        save_name = os.path.basename(neg_file).split(".")[0] + f"_thresh-{int(thresh * 100)}.npz"
        fname = f"{save_dir}/{save_name}"
        sp.save_npz(fname, neg_lbl)

    elif expt_no == 1:
        thresh = 0.7

        neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz"
        neg_lbl = sp.load_npz(neg_file)

        tneg_lbl = retain_topk(neg_lbl, k=5)

        thresh_file = f"/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-{int(thresh * 100)}.npz"
        thresh_lbl = sp.load_npz(thresh_file)
        thresh_lbl.data[:] = 1.0

        neg_lbl = neg_lbl.multiply(thresh_lbl)
        save_file = f"/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/ce-negatives_trn_X_Y_thresh-{int(thresh * 100)}.npz"
        sp.save_npz(save_file, neg_lbl)

        # positives

        gt_file = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
        gt_lbl = sp.load_npz(gt_file)

        pos_file = "/data/datasets/beir/msmarco/XC/trn_X_Y.npz"
        pos_lbl = sp.load_npz(pos_file).astype(np.float32)
        pos_lbl.data[:] = 0.0

        elidx2lidx = map_from_exact_to_full()
        rows, cols = gt_lbl.nonzero()
        cols = [elidx2lidx[c] for c in cols]
        pos_lbl[rows, cols] = gt_lbl.data

        elidx2lidx = map_from_negatives_to_full()
        rows, cols = tneg_lbl.nonzero()
        cols = [elidx2lidx[c] for c in cols]
        pos_lbl[rows, cols] = tneg_lbl.data

        save_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/ce-positives_trn_X_Y_top5.npz"
        sp.save_npz(save_file, pos_lbl)

    elif expt_no == 2:

        thresh = 0.8

        fname = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/trn_X_Y_normalize-exact.npz"
        pos_lbl = sp.load_npz(fname)

        pred_file = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/msmarco/train_predictions.npz"
        pred_lbl = sp.load_npz(pred_file)

        mask = np.hstack([n.data < p.max() * thresh for n,p in tqdm(zip(pred_lbl, pos_lbl), total=pos_lbl.shape[0])])
        pred_lbl.data[mask] = 0.0
        pred_lbl.eliminate_zeros()

        rows, cols = pos_lbl.nonzero()
        elidx2lidx = map_from_exact_to_full()
        cols = [elidx2lidx[c] for c in cols]
        pred_lbl[rows, cols] = pos_lbl.data

        save_dir = os.path.dirname(pred_file)
        save_name = "positives_trn_X_Y_normalize" + f"_thresh-{int(thresh * 100)}.npz"
        fname = f"{save_dir}/{save_name}"
        sp.save_npz(fname, pred_lbl)

    elif expt_no == 3:
        thresh = 0.9

        fname = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
        pos_lbl = sp.load_npz(fname)

        neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz"
        neg_lbl = sp.load_npz(neg_file)

        mask = np.hstack([n.data > p.max() * thresh for n,p in tqdm(zip(neg_lbl, pos_lbl), total=pos_lbl.shape[0])])
        neg_lbl.data[mask] = 0.0
        neg_lbl.eliminate_zeros()

        save_dir = os.path.dirname(neg_file)
        save_name = "ce-negatives-topk-05_trn_X_Y" + f"_thresh-{int(thresh * 100)}.npz"
        fname = f"{save_dir}/cross_encoder/{save_name}"
        sp.save_npz(fname, neg_lbl)

    elif expt_no == 4:
        thresh = 0.85

        fname = "/data/datasets/beir/msmarco/XC/trn_X_Y_ce-exact.npz"
        pos_lbl = sp.load_npz(fname)

        neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz"
        neg_lbl = sp.load_npz(neg_file)

        mask = np.hstack([n.data < p.max() * thresh for n,p in tqdm(zip(neg_lbl, pos_lbl), total=pos_lbl.shape[0])])
        neg_lbl.data[mask] = 0.0
        neg_lbl.eliminate_zeros()

        save_dir = os.path.dirname(neg_file)
        save_name = "ce-positives-topk-05_trn_X_Y" + f"_thresh-{int(thresh * 100)}.npz"
        fname = f"{save_dir}/cross_encoder/{save_name}"
        sp.save_npz(fname, neg_lbl)

