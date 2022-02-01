import torch
from torch import nn

import torch.nn.functional as F
from .modeling_IFA import IFAModel
from .clip.modeling_clip import CLIPModel
from .clip.configuration_clip import CLIPConfig
from transformers import BertConfig, BertModel

class IFAREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel, self).__init__()

        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config)

        # load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        self.model.resize_token_embeddings(len(tokenizer))

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        bsz = input_ids.size(0)
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits