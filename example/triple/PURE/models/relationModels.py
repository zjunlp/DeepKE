import os
import sys
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None):
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForRelationApprox(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        """
        attention_mask: [batch_size, from_seq_length, to_seq_length]
        """
        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits

class AlbertForRelationApprox(BertPreTrainedModel):
    """
    ALBERT approximation model is not supported by the current implementation, 
    as Huggingface's Transformers ALBERT doesn't support an attention mask with a shape of [batch_size, from_seq_length, to_seq_length]."
    """
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        batch_size = input_ids.size(0)
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits
