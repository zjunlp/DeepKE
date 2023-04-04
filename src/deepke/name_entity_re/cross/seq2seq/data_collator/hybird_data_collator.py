#!/usr/bin/env python
# coding=utf-8
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class HybirdDataCollator:
    data_collator_dict: Dict
    label_pad_token_id: int = -100
    meta_bucket_name: List[str] = None

    def data_group(self, features):
        bucket = defaultdict(list)
        for feature in features:
            task_name = feature['task']
            feature.pop('task')
            bucket[task_name] += [feature]
        return bucket

    def __call__(self, features) -> Dict[str, np.ndarray]:
        """ Hybird Data Collator

        Args:
            features (List[Dict[str, List]]):
                - input_ids
                - attention_mask

        Returns:
            [type]: [description]
        """

        pad_dict = {
            'input_ids': 0,
            'attention_mask': 0,
            'decoder_input_ids': 0,
            'labels': self.label_pad_token_id,
        }

        bucket = self.data_group(features)
        features = dict()
        for bucket_name, bucket_feature in bucket.items():
            # Pop unused feature;but meta-realted feature pop in DataCollatorForMetaSeq2Seq
            # Pop 无关的参数; Meta 任务不 Pop，在 DataCollatorForMetaSeq2Seq 中自动 Pop
            if self.meta_bucket_name is not None and bucket_name not in self.meta_bucket_name:
                for feature_name in list(bucket_feature[0].keys()):
                    if feature_name not in pad_dict:
                        [feature.pop(feature_name) for feature in bucket_feature]
            features[bucket_name] = self.data_collator_dict[bucket_name](bucket_feature)

        new_feature = dict()
        for feature_name, pad_value in pad_dict.items():
            sub_features = [sub_feature[feature_name] for sub_feature in features.values()]
            batch_size = sum([x.size(0) for x in sub_features])
            max_length = max([x.size(1) for x in sub_features])

            sub_feature = sub_features[0].new_full(
                size=(batch_size, max_length),
                fill_value=pad_value
            )
            start, end = 0, 0
            for x in sub_features:
                end = start + x.size(0)
                sub_feature[start:end, :x.size(1)] = x
                start = end
            new_feature[feature_name] = sub_feature

        return new_feature
