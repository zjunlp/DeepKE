import torch
from torch import nn
from torchcrf import CRF
import torch.nn.functional as F
from .modeling_IFA import IFAModel
from .clip.modeling_clip import CLIPModel
from .clip.configuration_clip import CLIPConfig
from transformers import BertConfig, BertModel

class IFANERCRFModel(nn.Module):
    def __init__(self, label_list, args):
        super(IFANERCRFModel, self).__init__()
        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config)

        self.num_labels  = len(label_list) + 1  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

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

        sequence_output = output.last_hidden_state       # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)             # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')  # 去掉CLS
            return logits, loss
        return logits, None