import argparse, torch.nn.functional as F, os, json, scipy.sparse as sp

from xcai.maggi.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dset_type', type=str, default="beir")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save_suffix', type=str, default=None)
    parser.add_argument('--phr_pred', action='store_true')
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    input_args = parse_args()

    # output_dir = "/home/sasokan/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    # output_dir = "/home/sasokan/b-sprabhu/outputs/mogicX/54_nvembed-for-msmarco-001/"
    # output_dir = "/data/suchith/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"
    output_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/"

    repr_dir = f"{output_dir}/representations/{input_args.dset_type}/{input_args.dataset}"

    if input_args.phr_pred: input_args.save_suffix = "phrase"
    suffix = "" if input_args.save_suffix is None else f"_{input_args.save_suffix}"

    # Load embeddings

    lbl_file = f"{repr_dir}/phr_repr.pth" if input_args.phr_pred else f"{repr_dir}/lbl_repr.pth"
    lbl_role = "phr" if input_args.phr_pred else "lbl"

    lbl_repr = combine_embeddings(lbl_file, lbl_role)
    lbl_repr = F.normalize(lbl_repr, dim=1) if input_args.normalize else lbl_repr

    tst_repr = combine_embeddings(f"{repr_dir}/tst_repr{suffix}.pth", "tst", suffix)
    tst_repr = F.normalize(tst_repr, dim=1) if input_args.normalize else tst_repr
    tst_lbl = None if input_args.phr_pred else sp.load_npz(f"/data/datasets/{input_args.dset_type}/{input_args.dataset}/XC/tst_X_Y.npz")

    if input_args.train:
        trn_repr = combine_embeddings(f"{repr_dir}/trn_repr{suffix}.pth", "trn", suffix)
        trn_repr = F.normalize(trn_repr, dim=1) if input_args.normalize else trn_repr
        trn_lbl = None if input_args.phr_pred else sp.load_npz(f"/data/datasets/{input_args.dset_type}/{input_args.dataset}/XC/trn_X_Y.npz")

    # Prediction

    pred_dir = f"{output_dir}/predictions/{input_args.dset_type}/{input_args.dataset}"
    os.makedirs(pred_dir, exist_ok=True)

    metrics, tst_pred = compute_metrics(tst_repr, lbl_repr, tst_lbl, metric_type="M")
    sp.save_npz(f"{pred_dir}/test_predictions{suffix}.npz", tst_pred)

    if input_args.train:
        m, trn_pred = compute_metrics(trn_repr, lbl_repr, trn_lbl, metric_type="M")
        sp.save_npz(f"{pred_dir}/train_predictions{suffix}.npz", trn_pred)
        if metrics is not None: metrics = {"train": m, "test": metrics}

    # Save metrics

    if metrics is not None:
        os.makedirs(f"{output_dir}/metrics/{input_args.dset_type}", exist_ok=True)
        metric_file = f"{output_dir}/metrics/{input_args.dset_type}/{input_args.dataset}{suffix}.json"
        with open(metric_file, "w") as file:
            json.dump(metrics, file, indent=4)

        print(metrics)

