import json
import os

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForTokenClassification, BertTokenizer
from collections import OrderedDict
from .BiLSTM_CRF import *

import hydra
from hydra import utils
import nltk
from nltk import word_tokenize


class BertNer(BertForTokenClassification):

    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        attention_mask=None, 
        valid_ids=None
    ):
        sequence_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        return logits

class InferNer:

    def __init__(self,model_dir: str, config, vocab_size=None, num_labels=None, word2id=None, id2label=None):
        """
        Custom Your Ner Model
        Args:
            vocab_size (`int`): (Only BiLSTM_CRF) vocabulary size used in BiLSTM_CRF
            num_labels (`int`): (Only BiLSTM_CRF)the number of Labels used in BiLSTM_CRF, such as the length of ([B-LOC, I-LOC, B-ORG, I-ORG, B-PER, I-PER, O])
            word2id (`dict`): (Only BiLSTM_CRF) Map word into index for embedding
            id2label (`dict`): (Only BiLSTM_CRF) Map index into Label for decoding
        """
        self.cfg = config
        if config.model_name == 'lstmcrf':
            assert vocab_size != None
            assert num_labels != None
            assert word2id != None
            assert id2label != None

            self.word2id = word2id
            self.id2label = id2label

            self.model = torch.load(os.path.join(utils.get_original_cwd(), self.cfg.output_dir, self.cfg.model_save_name))
        elif config.model_name == 'bert':
            self.model , self.tokenizer, self.model_config = self.load_model(model_dir)
            self.label_map = self.model_config["label_map"]
            self.max_seq_length = self.model_config["max_seq_length"]
            self.label_map = {int(k):v for k,v in self.label_map.items()}
        else:
            raise NotImplementedError(f"model type {self.cfg.model_name} not supported")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        model_config = os.path.join(model_dir,model_config)
        model_config = json.load(open(model_config))
        model = BertNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
        return model, tokenizer, model_config

    def tokenize(self, text: str):
        """ tokenize input"""
        if self.cfg.lan == "en":
            nltk.download('punkt')
            words = word_tokenize(text)
        else:
            words = list(text)
        tokens = []
        valid_positions = []
        for i,word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions

    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0,"[CLS]")
        valid_positions.insert(0,1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions

    def predict(self, text: str):
        if self.cfg.model_name == 'lstmcrf':
            words = [[self.word2id.get(w, 0) for w in text]]
            mask = [[1] * len(words[0])]
            y_pred = self.model(torch.tensor(words).to(self.device), torch.tensor(mask).bool().to(self.device))

            """Just adjust the format of output"""
            words = [w for w in text]
            labels = [(self.id2label[id], None) for id in y_pred[0]]
        elif self.cfg.model_name == 'bert':
            input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
            input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
            input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
            segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
            valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, valid_ids=valid_ids)
            logits = F.softmax(logits,dim=2)
            logits_label = torch.argmax(logits,dim=2)
            logits_label = logits_label.detach().cpu().numpy().tolist()[0]

            logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

            logits = []
            pos = 0
            for index,mask in enumerate(valid_ids[0]):
                if index == 0:
                    continue
                if mask == 1:
                    logits.append((logits_label[index-pos],logits_confidence[index-pos]))
                else:
                    pos += 1
            logits.pop()

            labels = [(self.label_map[label],confidence) for label,confidence in logits]
            if self.cfg.lan == "en":
                nltk.download('punkt')
                words = word_tokenize(text)
            else:
                words = list(text)
            assert len(labels) == len(words)
        else:
            raise NotImplementedError(f"model type {self.cfg.model_name} not supported")

        result = []
        for word, (label, confidence) in zip(words, labels):
            if label!='O':
                result.append((word,label))
        # tmp = []
        # tag = OrderedDict()
        # tag['PER'] = []
        # tag['LOC'] = []
        # tag['ORG'] = []
        
        # for i, (word, label) in enumerate(result):
        #     if label=='B-PER' or label=='B-LOC' or label=='B-ORG':
        #         if i==0:
        #             tmp.append(word)
        #         else:
        #             wordstype = result[i-1][1][2:]
        #             tag[wordstype].append(''.join(tmp))
        #             tmp.clear()
        #             tmp.append(word)
        #     else:
        #         tmp.append(word)
                
        #     if i==len(result)-1:
        #         if label=='B-PER' or label=='B-LOC' or label=='B-ORG':
        #             tmp.append(word)
        #         wordstype = result[i][1][2:]
        #         tag[wordstype].append(''.join(tmp))

        return result
