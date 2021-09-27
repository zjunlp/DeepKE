import numpy as np


class Seq2SeqSpanMetric(object):
    def __init__(self, eos_token_id, num_labels, target_type='word'):
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels+2

        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.em = 0
        self.total = 0
        self.target_type = target_type 

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1) # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1) # bsz
        target_seq_len = (target_seq_len-2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i]==target_seq_len[i]:
                em = int(tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item()==target_seq_len[i])
            self.em += em
            pairs = []
            cur_pair = []
            if len(ps):
                for j in ps:
                    if j<self.word_start_index:
                        if self.target_type == 'span':
                            if len(cur_pair)>0 and len(cur_pair)%2==0:
                                if all([cur_pair[i]<=cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                    pairs.append(tuple(cur_pair+[j]))
                        else:
                            if len(cur_pair) > 0:
                                if all([cur_pair[i]<cur_pair[i+1] for i in range(len(cur_pair)-1)]):
                                    pairs.append(tuple(cur_pair + [j]))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())

            tp, fn, fp = _compute_tp_fn_fp(pairs, ts)
            self.fn += fn
            self.tp += tp
            self.fp += fp

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.tp, self.fn, self.fp)
        res['f'] = round(f, 4)*100
        res['rec'] = round(rec, 4)*100
        res['pre'] = round(pre, 4)*100
        res['em'] = round(self.em/self.total, 4)
        if reset:
            self.total = 0
            self.fp = 0
            self.tp = 0
            self.fn = 0
            self.em = 0
        return res

def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, (set, list, np.ndarray)):
        ts = {tuple(key):1 for key in list(ts)}
    if isinstance(ps, (set, list, np.ndarray)):
        ps = {tuple(key):1 for key in list(ps)}

    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
