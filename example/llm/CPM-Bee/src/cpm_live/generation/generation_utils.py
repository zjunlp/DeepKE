import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits = logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1), descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits = logits.view(batch_size, -1).contiguous()

    return logits


def apply_repetition_penalty(
    logits,
    batch_size,
    num_beams,
    prev_output_tokens,
    repetition_penalty,
    start_idx=None,
    end_idx=None,
    window_size=None,
):
    # only conduct repetition penalty for the output
    assert repetition_penalty >= 1, "repetition penalty coefficient should >= 1"
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    for i in range(batch_size * num_beams):
        if start_idx is None or end_idx is None:
            output_tokens = prev_output_tokens[i].tolist()
        else:
            if end_idx >= start_idx:
                if window_size:
                    output_tokens = prev_output_tokens[i][
                        max(start_idx, end_idx + 1 - window_size) : end_idx + 1
                    ].tolist()
                else:
                    output_tokens = prev_output_tokens[i][start_idx : end_idx + 1].tolist()
            else:
                output_tokens = []
        for previous_token in set(output_tokens):
            # if score < 0 then repetition penalty has to
            # multiplied to reduce the previous token probability
            if logits[i, previous_token] < 0:
                logits[i, previous_token] *= repetition_penalty
            else:
                logits[i, previous_token] /= repetition_penalty


class BeamHypotheses:
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len**self.length_penalty
