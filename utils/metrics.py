import numpy as np

def dcg(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg(y_true, y_score, k=10):
    dcg_score = dcg(y_true, y_score, k)
    best_dcg = dcg(y_true, y_true, k)
    if best_dcg != 0:
        return  (float(dcg_score) / float(best_dcg))
    return dcg_score

def get_groups(qids):
        prev_qid = None
        prev_limit = 0
        total = 0

        for i, qid in enumerate(qids):
            total += 1
            if qid != prev_qid:
                if i != prev_limit:
                    yield (prev_qid, prev_limit, i)
                prev_qid = qid
                prev_limit = i

        if prev_limit != total:
            yield (prev_qid, prev_limit, total)

def check_qids(qids):
        seen_qids = set()
        prev_qid = None

        for qid in qids:
            assert qid is not None
            if qid != prev_qid:
                if qid in seen_qids:
                    raise ValueError('Samples must be grouped by qid.')
                seen_qids.add(qid)
                prev_qid = qid

        return len(seen_qids)


def queries_ndcg(y_true, y_score, qids, k = 10):
    query_groups = np.array([(qid, a, b, np.arange(a, b))
                                 for qid, a, b in get_groups(qids)],
                                dtype=np.object)

    n_queries = check_qids(qids)

    queries_ndcg = np.zeros(n_queries)

    for qidx, (qid, a, b, _) in enumerate(query_groups):
        # scores = model.predict(X.iloc[a:b])
        queries_ndcg[qidx] = ndcg(y_true.iloc[a:b], y_score[a:b], k)

    return queries_ndcg