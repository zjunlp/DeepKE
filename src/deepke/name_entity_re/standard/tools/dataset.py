import torch

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.
            Args:
                guid(string): Unique id for the example.
                text_a(string): The untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.
                text_b(string, optional): The untokenized text of the second sequence. Only must be specified for sequence pair tasks.
                label(string, optional): The label of the example. This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf-8')
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.strip().split()
        sentence.append(splits[0])
        label.append(splits[-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

def collate_fn(batch, word2id, label2id):
    batch.sort(key=lambda x: len(x.text_a.split(' ')), reverse=True)
    max_len = len(batch[0].text_a.split(' '))
    inputs = []
    targets = []
    masks = []

    UNK = word2id.get('<unk>')
    PAD = word2id.get('<pad>')
    for item in batch:
      input = item.text_a.split(' ')
      target = item.label.copy()

      input = [word2id.get(w, UNK) for w in input]
      target = [label2id.get(l) for l in target]
      pad_len = max_len - len(input)
      assert len(input) == len(target)
      inputs.append(input + [PAD] * pad_len)
      targets.append(target + [0] * pad_len)
      masks.append([1] * len(input) + [0] * pad_len)

    return torch.tensor(inputs), torch.tensor(targets), torch.tensor(masks).bool()
