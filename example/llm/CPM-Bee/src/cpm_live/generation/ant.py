import torch
import torch.nn.functional as F
from .generation_utils import BeamHypotheses, apply_repetition_penalty, top_k_top_p_filtering
from ..utils import pad


class CPMAntGeneration:
    def __init__(self, model, tokenizer, prompt_length=32):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def _convert_to_tensors(self, input_text, task_id=2):
        model_inputs = {}
        input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(input_text)
        input_ids = [j for j in input_ids if j != self.tokenizer.unk_id]

        model_inputs["input"] = [
            x + self.prompt_length * task_id for x in range(self.prompt_length)
        ] + input_ids
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = [0] * self.prompt_length + [2] * len(input_ids)

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

    def _process_texts(self, text_list):
        input_tensors = list(map(self._convert_to_tensors, text_list))
        keys = set(input_tensors[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(input_tensors, key, padding_side='left').cuda()
        return padded

    def generate(self, text_list, **kwargs):
        model_inputs = self._process_texts(text_list)
        with torch.inference_mode():
            result = self._decode(model_inputs, **kwargs)
        return result

    def _decode(self, model_inputs, **kwargs):
        raise NotImplementedError("_decode is not implemented.")


class CPMAntBeamSearch(CPMAntGeneration):
    def _decode(
        self,
        model_inputs,
        beam_size=3,
        max_length=100,
        repetition_penalty=1.0,
        repetition_window=None,
        **kwargs
    ):
        """
        Beam search
        Args:
            model_inputs (dict): input ids.
            beam_size (int, optional, defaults to 3): beam size of beam search.
            generate_length (int, optional, defaults to 100): maximum generation length.
            repetition_penalty (float, optional, defaults to 1.0): repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None): window size of repetition penalty, None means that all output tokens are penalized.
        """  # noqa: E501
        # generate_length + 1 for EOS token
        max_length += 1

        # expand dimmension
        batch_size = model_inputs["input"].size(0)
        input = (
            model_inputs["input"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        length = (
            model_inputs["length"]
            .unsqueeze(1)
            .expand(batch_size, beam_size)
            .contiguous()
            .view(
                batch_size * beam_size,
            )
        )
        context = (
            model_inputs["context"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        position = (
            model_inputs["position"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment = (
            model_inputs["segment"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        span = (
            model_inputs["span"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )

        done = [False for _ in range(batch_size)]

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_length, length_penalty=1, early_stopping=False)
            for _ in range(batch_size)
        ]

        pred_start_index = input.size(-1)
        past_key_values = None
        for i in range(max_length + 1):
            if i == 0:
                logits, _, past_key_values = self.model.inference(
                    input=input,
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )
            else:
                logits, _, past_key_values = self.model.inference(
                    input=input[:, -1:],
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )

            # skip all steps when we are done with each sentence
            if all(done):
                break

            # (batch * beam, seqlen, model_dim)
            logits = logits[:, -1, :]

            if i == 0:
                logits[:, self.tokenizer.eos_id] = -float("inf")
                logits[:, self.tokenizer.newline_id] = -float("inf")

            apply_repetition_penalty(
                logits,
                batch_size,
                beam_size,
                input,
                repetition_penalty,
                pred_start_index,
                input.size(-1) - 1,
                repetition_window,
            )
            scores = F.log_softmax(logits, dim=-1)

            next_scores = scores + beam_scores[:, None].expand_as(
                scores
            )  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis across beams)
            next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )

            assert next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)
            next_batch_beam = []

            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item(), i
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.tokenizer.pad_id, 0)] * beam_size
                    )  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor")
                    word_id = idx % scores.size(-1)

                    # end of sentence, or next word
                    if word_id == self.tokenizer.eos_id or i == max_length:
                        generated_hyps[sent_id].add(
                            input[sent_id * beam_size + beam_id, pred_start_index:]
                            .clone()
                            .cpu()
                            .tolist(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if i == max_length else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.tokenizer.pad_id, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # we have reached the last step
            if i == max_length:
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input.new([x[1] for x in next_batch_beam])
            beam_idx = length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input = input[beam_idx, :]

            past_key_values = [list(each) if each is not None else each for each in past_key_values]  # type: ignore # noqa: E501
            for key_value_layer in past_key_values:
                if key_value_layer is not None:
                    key_value_layer[0] = key_value_layer[0][beam_idx]
                    key_value_layer[1] = key_value_layer[1][beam_idx]

            # update input ids
            input = torch.cat([input, beam_words.unsqueeze(1)], dim=-1)
            length += 1
            context = torch.cat(
                [context, torch.ones((context.size(0), 1), dtype=torch.int, device=context.device)],
                dim=-1,
            )
            position = torch.cat([position, position[:, -1:] + 1], dim=-1)
            segment = torch.cat(
                [segment, segment[:, -1:]], dim=-1
            )  # segment id always the same as the previous token
            span = torch.cat([span, span[:, -1:]], dim=-1)

        # select the best hypotheses
        results = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            results.append(best_hyp)

        result_text = list(map(self.tokenizer.decode, results))
        return result_text


class CPMAntRandomSampling(CPMAntGeneration):
    def _decode(
        self,
        model_inputs,
        max_length=100,
        top_k=0,
        top_p=0.9,
        temperature=0.9,
        repetition_penalty=1.0,
        repetition_window=None,
        **kwargs
    ):
        """
        Top-k and top-p sampling.
        Args:
            model_inputs (dict): input ids
            generate_length (int, optional, defaults to 100): maximum generation length
            top_k (int, optional, defaults to 0): keep only top k tokens with highest probability. 0 means keeping all tokens.
            top_p (int, optional, defaults to 0.9): keep the top tokens with cumulative probability >= top_p.
            temperature (int, optional, defaults to 0.9): the value that can cool down the logits distribution.
            repetition_penalty (float, optional, defaults to 1.0): repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None): window size of repetition penalty, None means that all output tokens are penalized.
        """  # noqa: E501
        # generate_length + 1 for EOS token
        max_length += 1

        input = model_inputs["input"]
        length = model_inputs["length"]
        context = model_inputs["context"]
        position = model_inputs["position"]
        segment = model_inputs["segment"]
        span = model_inputs["span"]
        batch_size = input.size(0)

        pred_start_index = input.size(-1)
        past_key_values = None
        done = [False for _ in range(batch_size)]
        results = [None for _ in range(batch_size)]
        for i in range(max_length):
            if i == 0:
                logits, _, past_key_values = self.model.inference(
                    input=input,
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )
            else:
                logits, _, past_key_values = self.model.inference(
                    input=input[:, -1:],
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )

            logits = logits[:, -1, :]

            if i == 0:
                logits[:, self.tokenizer.eos_id] = -float("inf")
                logits[:, self.tokenizer.newline_id] = -float("inf")

            apply_repetition_penalty(
                logits,
                batch_size,
                1,
                input,
                repetition_penalty,
                pred_start_index,
                input.size(-1) - 1,
                repetition_window,
            )

            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            for idx in range(batch_size):
                if not done[idx] and (
                    next_token[idx].item() == self.tokenizer.eos_id or i == max_length - 1
                ):
                    done[idx] = True
                    results[idx] = input[idx, pred_start_index:].clone().cpu().tolist()  # type: ignore # noqa: E501

            if sum(done) == batch_size:
                break

            # update input ids
            input = torch.cat([input, next_token], dim=-1)
            length += 1
            context = torch.cat(
                [context, torch.ones((context.size(0), 1), dtype=torch.int, device=context.device)],
                dim=-1,
            )
            position = torch.cat([position, position[:, -1:] + 1], dim=-1)
            segment = torch.cat(
                [segment, segment[:, -1:]], dim=-1
            )  # segment id always the same as the previous token
            span = torch.cat([span, span[:, -1:]], dim=-1)

        result_text = list(map(self.tokenizer.decode, results))
        return result_text
