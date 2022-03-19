import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from ..models.clip.processing_clip import CLIPProcessor
import logging
logger = logging.getLogger(__name__)


class MMREProcessor(object):
    def __init__(self, data_path, re_path, args):
        self.args = args
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})

        self.clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.aux_processor.feature_extractor.size, self.aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size
        self.rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        self.rcnn_processor.feature_extractor.size, self.rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size


    def load_from_file(self, mode="train"):
        load_file = os.path.join(self.args.cwd,self.data_path[mode])
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h']) # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        # 辅助图像
        aux_imgs = None
        # if not self.use_clip_vit:
        aux_path = os.path.join(self.args.cwd,self.data_path[mode+"_auximgs"])
        aux_imgs = torch.load(aux_path)
        rcnn_imgs = torch.load(os.path.join(self.args.cwd,self.data_path[mode+'_img2crop']))
        return {'words':words, 'relations':relations, 'heads':heads, 'tails':tails, 'imgids': imgids, 'dataid': dataid, 'aux_imgs':aux_imgs, "rcnn_imgs":rcnn_imgs}


    def get_relation_dict(self):
        
        with open(os.path.join(self.args.cwd,self.re_path), 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    def get_rel2id(self, train_path):
        with open(os.path.join(self.args.cwd,self.re_path), 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key:[] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MMREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, mode="train") -> None:
        self.processor = processor
        self.args = self.processor.args
        self.transform = transform
        self.max_seq = self.args.max_seq
        self.img_path = img_path[mode]  if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.aux_size = self.args.aux_size
        self.rcnn_size = self.args.rcnn_size
    
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if  i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)   # list不会进行子词分词
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)
        
        re_label = self.re_dict[relation]   # label to id

         # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(os.path.join(self.args.cwd,self.img_path), imgid)
                img_path = img_path.replace('test', 'train')
                image = Image.open(img_path).convert('RGB')
                image = self.processor.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(os.path.join(self.args.cwd,self.img_path), 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.processor.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # 辅助图像
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][item_id]
                    # print(aux_img_paths)
                    aux_img_paths = [os.path.join(os.path.join(self.args.cwd,self.aux_img_path), path) for path in aux_img_paths]
                # 大于3需要舍弃
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.processor.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                #小于3需要加padding-0
                for i in range(3-len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size))) 

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(os.path.join(self.args.cwd,self.rcnn_img_path), path) for path in rcnn_img_paths]
                     # 大于3需要舍弃
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.processor.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)
                    #小于3需要加padding-0
                    for i in range(3-len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size))) 

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, rcnn_imgs

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs
            
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)
    


