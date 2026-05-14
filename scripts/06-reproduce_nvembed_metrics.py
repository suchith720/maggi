import scipy.sparse as sp, pytrec_eval, torch
from beir.retrieval.evaluation import EvaluateRetrieval

from collections import defaultdict

from sugar.core import *
from xcai.metrics import *

if __name__ == "__main__":

    # datasets = ["arguana", "fiqa", "msmarco", "nfcorpus", "scidocs", "scifact", "trec-covid", "webis-touche2020"]

    datasets = ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact", "trec-covid", "webis-touche2020"]

    for dataset in datasets:
        print(dataset)

        tst_ids, tst_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/test.raw.csv")
        lbl_ids, lbl_txt = load_raw_file(f"/data/datasets/beir/{dataset}/XC/raw_data/label.raw.csv")
        tst_lbl = sp.load_npz(f"/data/datasets/beir/{dataset}/XC/tst_X_Y.npz")
        # tst_lbl.data[:] = 1.0
        qrels = {str(i): {str(lbl_ids[p]):int(q) for p,q in zip(r.indices, r.data)} for i,r in zip(tst_ids, tst_lbl)}

        tst_pred = sp.load_npz(f"/data/outputs/maggi/00_nvembed-to-compute-msmarco-embeddings-001/predictions/beir/{dataset}/test_labels.npz")
        results = {str(i): {str(lbl_ids[p]):float(q) for p,q in zip(r.indices, r.data)} for i,r in zip(tst_ids, tst_pred)}

        ## beir evaluator

        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10, 100])
        print(ndcg)

        ## pytrec_eval

        metrics = {"map", "ndcg_cut.10", "recall.10", "P.10"}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
        scores = evaluator.evaluate(results)

        avg = defaultdict(float)
        for query_scores in scores.values():
            for metric, value in query_scores.items():
                avg[metric] += value
        for metric in avg:
            avg[metric] /= len(scores)
        print(dict(avg))

        ## my code

        # metric = BeirMetric(len(lbl_ids), k_values=[1, 3, 5, 10], qry_ids=tst_ids, lbl_ids=lbl_ids)
        metric = PrecReclHits(len(lbl_ids), pk=10, rk=200, hk=10, rep_pk=[1, 3, 5, 10], rep_rk=[5, 10, 100, 200], rep_hk=[1, 3, 5, 10])
        o = {
            'pred_idx': torch.tensor(tst_pred.indices, dtype=torch.int64),
            'pred_score': torch.tensor(tst_pred.data, dtype=torch.float32),
            'pred_ptr': torch.tensor([p-q for p,q in zip(tst_pred.indptr[1:], tst_pred.indptr)], dtype=torch.int64),
        }

        assert tst_lbl.shape == tst_pred.shape
        t = {
            'targ_idx': torch.tensor(tst_lbl.indices, dtype=torch.int64),
            'targ_score': torch.tensor(tst_lbl.data, dtype=torch.float32),
            'targ_ptr': torch.tensor([p-q for p,q in zip(tst_lbl.indptr[1:], tst_lbl.indptr)], dtype=torch.int64),
        }
        value = metric(**o, **t)
        print(value)

        print()
