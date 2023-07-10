from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple


def preprocess_cpmbee(example, prompter, tokenizer, options):
    #data = {"prompt": example["instruction"], "input": example["input"], "<ans>": example["output"]}
    data = {"input": example["instruction"]+ "\n" + example["input"], "<ans>": example["output"]}
    raw_data = {}
    (
        input_ids,
        input_id_subs,
        context,
        segment_ids,
        segment_rel,
        n_segments,
        _
    ) = tokenizer.convert_data_to_id(data)
    input_ids = input_ids[: options.cutoff_len]
    input_id_subs = input_id_subs[: options.cutoff_len]
    context = context[: options.cutoff_len]
    segment_ids = segment_ids[: options.cutoff_len]
    raw_data["input"] = data
    raw_data["samples"] = []
    sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
    segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
    num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)

    return {"input_ids": input_ids,  "inputs_sub": input_id_subs,  "context": context,  "sample_ids": sample_ids,  "segments": segment_ids,  "num_segments": num_segments,  "segment_rel_offset": segment_rel_offset,  "segment_rel": segment_rel,  "spans": [input_ids.shape[0]],  "raw_data": raw_data}


def coll_fn_cpmbee(stage = "sft"):
    return preprocess_cpmbee



@dataclass
class DataCollatorForCPMBEE:
    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(self, features):
        _inputs: List[NDArray[np.int32]] = []
        _inputs_sub: List[NDArray[np.int32]] = []
        _context: List[NDArray[np.int8]] = []
        _sample_ids: List[NDArray[np.int32]] = []
        _segments: List[NDArray[np.int32]] = []
        _num_segments: List[NDArray[np.int32]] = []
        _segment_rel_offset: List[NDArray[np.int32]] = []
        _segment_rel: List[NDArray[np.int32]] = []
        _spans: List[List[int]] = []
        _raw_data: List[List[Any]] = []

        for feature in features:
            _inputs.append(np.array(feature["input_ids"], dtype=np.int32))
            _inputs_sub.append(np.array(feature["inputs_sub"], dtype=np.int32))
            _context.append(np.array(feature["context"], dtype=np.int8))
            _sample_ids.append(np.array(feature["sample_ids"], dtype=np.int32))
            _segments.append(np.array(feature["segments"], dtype=np.int32))
            _num_segments.append(np.array(feature["num_segments"], dtype=np.int32))
            _segment_rel_offset.append(np.array(feature["segment_rel_offset"], dtype=np.int32))
            _segment_rel.append(np.array(feature["segment_rel"], dtype=np.int32))
            _spans.append(feature["spans"])
            _raw_data.append(feature["raw_data"])

        batch_size = len(_inputs)
        inputs = np.zeros((batch_size, self.max_length), dtype=np.int32)
        inputs_sub = np.zeros((batch_size, self.max_length), dtype=np.int32)
        context = np.zeros((batch_size, self.max_length), dtype=np.int8)
        sample_ids = np.zeros((batch_size, self.max_length), dtype=np.int32)
        segments = np.zeros((batch_size, self.max_length), dtype=np.int32)
        num_segments = np.zeros((batch_size, self.max_length), dtype=np.int32)
        segment_rel_offset = np.zeros((batch_size, self.max_length), dtype=np.int32)
        tgt = np.full((batch_size, self.max_length), -100, dtype=np.int32)

        max_rel = 0
        for i in range(batch_size):
            max_rel = max(max_rel, _segment_rel[i].shape[0])
        segment_rel = np.zeros((batch_size, max_rel), dtype=np.int32)
        spans = np.zeros((batch_size, self.max_length), dtype=np.int32)
        length = np.zeros((batch_size,), dtype=np.int32)

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []
        raw_data_list: List[Any] = []

        for i in range(batch_size):
            instance_length = _inputs[i].shape[0]
            rel_size = _segment_rel[i].shape[0]
            inputs[i, :instance_length] = _inputs[i]
            inputs_sub[i, :instance_length] = _inputs_sub[i]
            context[i, :instance_length] = _context[i]
            sample_ids[i, :instance_length] = _sample_ids[i]
            segments[i, :instance_length] = _segments[i]
            num_segments[i, :instance_length] = _num_segments[i]
            segment_rel_offset[i, :instance_length] = _segment_rel_offset[i]
            segment_rel[i, :rel_size] = _segment_rel[i]

            span_begin = 0
            for span_id, span_end in enumerate(_spans[i]):
                spans[i, span_begin:span_end] = span_id
                span_begin = span_end
            length[i] = instance_length
            raw_data_list.extend(_raw_data[i])

            for j in range(instance_length):
                idx, idx_sub = _inputs[i][j], _inputs_sub[i][j]
                tgt_idx = idx
                if idx_sub > 0:
                    # need to be in ext table
                    if (idx, idx_sub) not in batch_ext_table_map:
                        batch_ext_table_map[(idx, idx_sub)] = len(batch_ext_table_map)
                        batch_ext_table_ids.append(idx)
                        batch_ext_table_sub.append(idx_sub)
                    tgt_idx = batch_ext_table_map[(idx, idx_sub)] + self.tokenizer.vocab_size
                if j > 1 and context[i, j - 1] == 0:
                    if idx != self.tokenizer.bos_token_id:
                        tgt[i, j - 1] = tgt_idx
                    else:
                        tgt[i, j - 1] = self.tokenizer.eos_token_id
            if context[i, instance_length - 1] == 0:
                tgt[i, instance_length - 1] = self.tokenizer.eos_token_id
        
        if len(batch_ext_table_map) == 0:
            # placeholder
            batch_ext_table_ids.append(0)
            batch_ext_table_sub.append(1)

        return BatchEncoding({
            "input_ids": inputs,
            "input_id_sub": inputs_sub,
            "length": length,
            "context": context > 0,
            "sample_ids": sample_ids,
            "num_segments": num_segments,
            "segment": segments,
            "segment_rel_offset": segment_rel_offset,
            "segment_rel": segment_rel,
            "span": spans,
            "labels": tgt,
            "ext_table_ids": np.array(batch_ext_table_ids, dtype=np.int32),
            "ext_table_sub": np.array(batch_ext_table_sub, dtype=np.int32)
        }, tensor_type="pt")

