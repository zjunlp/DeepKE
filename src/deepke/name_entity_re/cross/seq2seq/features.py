#!/usr/bin/env python
# -*- coding:utf-8 -*-
from datasets import Features, Value, Sequence

DatasetFeature = Features({
    'text': Value(dtype='string', id=None),
    'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'record': Value(dtype='string', id=None),
    'entity': [{'type': Value(dtype='string', id=None),
                'offset': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
                'text': Value(dtype='string', id=None)}],
    'relation': [{'type': Value(dtype='string', id=None),
                  'args': [{'type': Value(dtype='string', id=None),
                            'offset': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
                            'text': Value(dtype='string', id=None)}]}],
    'event': [{'type': Value(dtype='string', id=None),
               'offset': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
               'text': Value(dtype='string', id=None),
               'args': [{'type': Value(dtype='string', id=None),
                         'offset': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
                         'text': Value(dtype='string', id=None)}]}],
    'spot': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'asoc': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'spot_asoc': [{'span': Value(dtype='string', id=None),
                   'label': Value(dtype='string', id=None),
                   'asoc': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None)}],
    'task': Value(dtype='string', id=None),
})


_processed_feature = {
    'input_ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
    'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
    'labels': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
    'spots': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'asocs': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'spot_asoc': [
        {'span': Value(dtype='string', id=None),
         'label': Value(dtype='string', id=None),
         'asoc': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None)}
    ],
    'task': Value(dtype='string', id=None),
    'sample_prompt': Value(dtype='bool', id=None)
}


ProcessedFeature = Features(_processed_feature)


RecordFeature = Features({
    'input_ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
    'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
    'labels': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
    'spots': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'asocs': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'spot_asoc': [
        {'span': Value(dtype='string', id=None),
         'label': Value(dtype='string', id=None),
         'asoc': Sequence(feature=Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), length=-1, id=None)}
    ],
    'sample_prompt': Value(dtype='bool', id=None)
})
