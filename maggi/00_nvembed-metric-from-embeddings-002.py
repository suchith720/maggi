import argparse, torch.nn.functional as F, os, json, scipy.sparse as sp, numpy as np

from sugar.core import *
from xcai.maggi.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dset_type', type=str, default="beir")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--normalize', action='store_true')

    parser.add_argument('--meta_pred', action='store_true')
    parser.add_argument('--meta_name', type=str, default=None)
    parser.add_argument('--meta_role', type=str, default=None)

    parser.add_argument('--similarity', action='store_true')
    parser.add_argument('--related_query', action='store_true')
    parser.add_argument('--exact', action='store_true')

    parser.add_argument('--lbl_rep_file', type=str, default=None)
    parser.add_argument('--meta_rep_file', type=str, default=None)

    parser.add_argument('--repr_suffix', type=str, default=None)
    parser.add_argument('--save_suffix', type=str, default=None)
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    input_args = parse_args()

    """
    Setting `variables`: output_dir, data_dir, repr_dir, metric_dir, repr_suffix, save_suffix

    """

    output_dir = "/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-004/"
    data_dir = f"/data/datasets/{input_args.dset_type}/{input_args.dataset}/XC/"

    repr_dir = f"{output_dir}/representations/{input_args.dset_type}/{input_args.dataset}"
    metric_dir = f"{output_dir}/metrics/{input_args.dset_type}"

    if input_args.meta_role is None: input_args.meta_role = input_args.meta_name

    repr_suffix = "" if input_args.repr_suffix is None else f"_{input_args.repr_suffix}"
    save_suffix = "" if (input_args.save_suffix is None or input_args.meta_pred) else f"-{input_args.save_suffix}"

    # Load embeddings

    if input_args.related_query:
        """
        Get related queries

        """

        tst_repr = combine_embeddings(f"{repr_dir}/tst_repr{repr_suffix}.pth", "tst", repr_suffix)
        tst_repr = F.normalize(tst_repr, dim=1) if input_args.normalize else tst_repr

        trn_repr = combine_embeddings(f"{repr_dir}/trn_repr{repr_suffix}.pth", "trn", repr_suffix)
        trn_repr = F.normalize(trn_repr, dim=1) if input_args.normalize else trn_repr

        pred_dir = f"{output_dir}/predictions/{input_args.dset_type}/{input_args.dataset}"
        os.makedirs(pred_dir, exist_ok=True)

        _, tst_trn = compute_metrics(tst_repr, trn_repr)
        sp.save_npz(f"{pred_dir}/test_train_query.npz", tst_trn)

        if input_args.train:
            _, trn_trn = compute_metrics(trn_repr, trn_repr)
            sp.save_npz(f"{pred_dir}/train_train_query.npz", trn_trn)
    else:

        if input_args.meta_pred:
            lbl_file, lbl_role, lbl_name = f"{repr_dir}/{input_args.meta_name}_repr.pth", input_args.meta_role, input_args.meta_name
            if input_args.meta_rep_file is not None: lbl_file = input_args.meta_rep_file
        else:
            lbl_file, lbl_role, lbl_name = f"{repr_dir}/lbl_repr.pth", "lbl", "labels"
            if input_args.lbl_rep_file is not None: lbl_file = input_args.lbl_rep_file

        lbl_repr = combine_embeddings(lbl_file, lbl_role)
        lbl_repr = F.normalize(lbl_repr, dim=1) if input_args.normalize else lbl_repr

        if input_args.similarity:
            pred_dir = f"{output_dir}/predictions/{input_args.dset_type}/{input_args.dataset}"
            _, lbl_lbl = compute_metrics(lbl_repr, lbl_repr)
            sp.save_npz(f"{pred_dir}/{lbl_name}_{lbl_name}.npz", lbl_lbl)

        else:
            tst_lbl = tst_repr = trn_lbl = trn_repr = None

            tst_repr = combine_embeddings(f"{repr_dir}/tst_repr{repr_suffix}.pth", "tst", repr_suffix)
            tst_repr = F.normalize(tst_repr, dim=1) if input_args.normalize else tst_repr

            if input_args.meta_pred: 
                tst_lbl = tst_ids = lbl_ids = None

            else:
                tst_lbl = sp.load_npz(f"{data_dir}/tst_X_Y.npz")
                tst_ids, tst_txt = load_raw_file(f"{data_dir}/raw_data/test.raw.csv")
                lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")

            if input_args.train:
                trn_repr = combine_embeddings(f"{repr_dir}/trn_repr{repr_suffix}.pth", "trn", repr_suffix)
                trn_repr = F.normalize(trn_repr, dim=1) if input_args.normalize else trn_repr

                if input_args.meta_pred:
                    trn_lbl = trn_ids = None
                else:
                    trn_lbl = sp.load_npz(f"{data_dir}/trn_X_Y.npz")
                    trn_ids, trn_txt = load_raw_file(f"{data_dir}/raw_data/train.raw.csv")
                    if lbl_ids is None: lbl_ids, lbl_txt = load_raw_file(f"{data_dir}/raw_data/label.raw.csv")

            if input_args.exact and (not input_args.meta_pred):
                if trn_repr is None: trn_lbl = sp.load_npz(f"{data_dir}/trn_X_Y.npz")
                nnz = trn_lbl.getnnz(axis=0) + tst_lbl.getnnz(axis=0)
                valid_idxs = np.where(nnz > 0)[0]

                tst_lbl = tst_lbl[:, valid_idxs]
                if trn_repr is not None: trn_lbl = trn_lbl[:, valid_idxs]
                lbl_repr = lbl_repr[valid_idxs]
                lbl_name = f"{lbl_name}-exact"
                repr_suffix = f"{repr_suffix}-exact"

            # Prediction

            pred_dir = f"{output_dir}/predictions/{input_args.dset_type}/{input_args.dataset}"
            os.makedirs(pred_dir, exist_ok=True)

            metrics, tst_pred = compute_metrics(tst_repr, lbl_repr, tst_lbl, qry_ids=tst_ids, lbl_ids=lbl_ids)
            sp.save_npz(f"{pred_dir}/test{save_suffix}_{lbl_name}.npz", tst_pred)

            if input_args.train:
                m, trn_pred = compute_metrics(trn_repr, lbl_repr, trn_lbl, qry_ids=trn_ids, lbl_ids=lbl_ids)
                sp.save_npz(f"{pred_dir}/train{save_suffix}_{lbl_name}.npz", trn_pred)
                if metrics is not None: metrics = {"train": m, "test": metrics}

            # Save metrics

            if metrics is not None:
                os.makedirs(metric_dir, exist_ok=True)
                metric_file = f"{metric_dir}/{input_args.dataset}{repr_suffix}.json"
                with open(metric_file, "w") as file:
                    json.dump(metrics, file, indent=4)

                print(metrics)

