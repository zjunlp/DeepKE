import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from ..models.clip.processing_clip import CLIPProcessor

import logging
logger = logging.getLogger(__name__)


class MMPNERProcessor(object):
    def __init__(self, data_path, args) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_name, do_lower_case=True)
        self.clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor.feature_extractor.size, self.aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size
        self.rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.rcnn_processor.feature_extractor.size, self.rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size


    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs))
        aux_imgs = None
        # if not self.use_clip_vit:
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

        rcnn_imgs = torch.load(self.data_path['img2crop'])

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs, "rcnn_imgs":rcnn_imgs}


class MMPNERDataset(Dataset):
    def __init__(self, processor, label_mapping, img_path=None, aux_path=None, rcnn_img_path=None, max_seq=40, ignore_idx=-100, aux_size=128, rcnn_size=64, mode='train') -> None:
        self.processor = processor
        self.data_dict = processor.load_from_file(mode)
        self.tokenizer = processor.tokenizer
        self.label_mapping = label_mapping
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_path[mode]  if aux_path is not None else None
        self.rcnn_img_path = rcnn_img_path
        self.mode = mode
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
    
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                for i in range(3-len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    img = img.split('.')[0]
                    if img in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][img]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    for i in range(3-len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size))) 

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs, rcnn_imgs

                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_imgs

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)