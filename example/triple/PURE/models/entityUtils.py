import numpy as np
import json
import logging

logger = logging.getLogger('root')

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

def convert_dataset_to_samples(dataset, max_span_length, ner_label2id=None, context_window=0, split=0):
    """
    Extract sentences and gold entities from a dataset
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0
    
    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset)*0.9))
    elif split == 2:
        data_range = (int(len(dataset)*0.9), len(dataset))

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            num_ner += len(sent.ner)
            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }
            if context_window != 0 and len(sent.text) > context_window:
                logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
                # print('Exclude:', sample)
                # continue
            sample['tokens'] = sent.text
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left
                
                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start
            
            sent_ner = {}
            for ner in sent.ner:
                sent_ner[ner.span.span_sent] = ner.label

            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []
            for i in range(len(sent.text)):
                for j in range(i, min(len(sent.text), i+max_span_length)):
                    sample['spans'].append((i+sent_start, j+sent_start, j-i+1))
                    span2id[(i, j)] = len(sample['spans'])-1
                    if (i, j) not in sent_ner:
                        sample['spans_label'].append(0)
                    else:
                        sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])
            samples.append(sample)
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('# Overlap: %d'%num_overlap)
    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length'%(len(samples), data_range[1]-data_range[0], num_ner, avg_length, max_length))
    logger.info('Max Length: %d, max NER: %d'%(max_len, max_ner))
    return samples, num_ner

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
