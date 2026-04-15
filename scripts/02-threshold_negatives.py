import numpy as np, scipy.sparse as sp, os
from tqdm.auto import tqdm

if __name__ == "__main__":
    expt_no = 0

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

        thresh = 0.9

        neg_file = "/data/datasets/beir/msmarco/XC/ce-negatives-topk-05_trn_X_Y.npz"
        neg_lbl = sp.load_npz(neg_file)

        thresh_file = f"/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/negatives_trn_X_Y_normalize_thresh-{int(thresh * 100)}.npz"
        thresh_lbl = sp.load_npz(thresh_file)
        thresh_lbl.data[:] = 1.0

        neg_lbl = neg_lbl.multiply(thresh_lbl)
        save_file = "/data/outputs/mogicX/54_nvembed-for-msmarco-001/matrices/msmarco/ce-negatives_trn_X_Y_normalize_thresh-{int(thresh * 100)}.npz"
        sp.save_npz(save_file, neg_lbl)

