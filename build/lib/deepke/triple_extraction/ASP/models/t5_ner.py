"""
    Autoregressive Structured Prediction T5 model for Named Entity Recognition
    Tianyu Liu
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ctypes import sizeof
import math
from typing import Optional, Tuple, Any, Dict, Iterable, List
import logging

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel, T5Model,
    T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
)
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)

from .modeling_outputs import MySeq2SeqLMOutput

import util
from util import (
    batched_masked_select,
    logsumexp
)
logger = logging.getLogger()
NEGINF = -20000.

class T5NER(T5PreTrainedModel):
    base_model_prefix = "t5"

    def __init__(
        self,
        config,
        asp_hidden_dim: int = 1500,
        asp_dropout_rate: float = 0.3,
        asp_init_std: float = 0.02,
        asp_feature_emb_size: int = 20,
        asp_linking_distance_num_buckets: int = 16,
        asp_activation: str='relu',
        num_typing_classes: int = 4,
        mention_start_id: int = 0,
        mention_end_id: int = 0,
    ):
        super().__init__(config)
        self.t5 = T5Model.from_pretrained(self.config._name_or_path)
        self.dropout = nn.Dropout(p=asp_dropout_rate)
        self.init_std = asp_init_std

        self.feature_emb_size = asp_feature_emb_size
        self.linking_distance_num_buckets = asp_linking_distance_num_buckets
        # output_dim: {0, 1, 2}
        # corresponding to 3 actions: {mention_end, mention_start, copy}

        self.num_typing_classes = num_typing_classes
        # conll04, ace04, ace05 has no nested mention
        self.max_nest_depth = 1
        self.action_head = util.make_ffnn(
            self.config.d_model,
            hidden_size=[asp_hidden_dim],
            output_size=1,
            dropout=self.dropout,
            std=self.init_std,
            activation=asp_activation
        )

        lr_scorer_input_dim = 2*self.config.d_model
        # left-right bracket
        self.lr_scorer = util.make_ffnn(
            lr_scorer_input_dim,
            hidden_size=[asp_hidden_dim],
            output_size=self.num_typing_classes,
            dropout=self.dropout,
            std=self.init_std,
            activation=asp_activation
        )
        self.last_device = torch.cuda.device_count() - 1

        self.mention_start_id = mention_start_id
        self.mention_end_id = mention_end_id
        self.post_init()


    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MySeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        sentence_map=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_input_ids=None,
        full_decoder_input_ids=None,
        full_hidden_states=None,
        lr_pair_flag=None,
        sentence_idx=None,
        decoder_pairing=None,
        decoder_typing=None,
    ):
        r"""
        Inputs:
            labels (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*, *training*):
                Labels for computing the masked language modeling loss. 
                Indices should either be in `[0, 1, 2]` (3 basic actions) or -1 (see `input_ids` docstring). 
                labels with <0 values are ignored (masked).
            lr_pair_flag (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_l_brackets)`, *optional*, *training*):
                A flag tensor. Indicating whether a pair of tokens are paired *left-right* brackts. 1 for paired, 0 for not.
            rr_pair_flag (`torch.LongTensor` of shape `(batch_size, sequence_length, number_of_r_brackets)`, *optional*, *training*):
                A flag tensor. Indicating whether a pair of tokens are paired *right-right* brackts. 1 for paired, 0 for not.
            full_decoder_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, *inference*):
                Decoded token ids until now. Keeping track of how many tokens have been copied from
                the input, so we know which token to copy now.
            full_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, *inference*):
                Hidden states of the decoded tokens until now. 
            decoder_pairing (`List[torch.LongTensor]` each element is shape `(batch_size, 1)`, *optional*, *inference*):
                A list of tensors. Each tensor is a tensor of shape `(batch_size, 1)`, decoded pairing actions until now.
            decoder_typing (`List[torch.LongTensor]` each element is shape `(batch_size, 1)`, *optional*, *inference*):
                A list of tensors. Each tensor is a tensor of shape `(batch_size, 1)`, decoded typing actions until now.
        Returns:
            output (`MySeq2SeqLMOutput` ):
                loss (`torch.FloatTensor` of shape `(1)`, *optional*, returned when `labels` is provided):
                pairing (`torch.LongTensor` of shape `(batch_size, 1)`, pairing actions of this step, *inferece* only)
                typing (`torch.LongTensor` of shape `(batch_size, 1)`, typing actions of this step, *inferece* only)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        target_ids = decoder_input_ids
        if labels is not None:  # training
            # decoder_input_ids starts with <pad> and has the same length as target_ids
            decoder_input_ids = self._shift_right(
                target_ids
            )

        outputs = self.t5(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss, lm_logits = None, None
        if labels is not None:  # Training: We compute loss
            outputs.last_hidden_state = outputs.last_hidden_state.to(self.device)

            # shape: (batchsize, seq_len, 1)
            action_logits = self.action_head(outputs.last_hidden_state)

            # a new loss that is compatible with inference
            (numer, denom) = self.get_logits_training(
                outputs.last_hidden_state,  # current_hidden_state
                target_ids=target_ids, 
                lr_pair_flag=lr_pair_flag,
                labels=labels
            )
            # keeping <copy> score 0.
            action_logits = torch.cat(
                [torch.zeros_like(action_logits),
                 action_logits], dim=-1
            )
            # Note: We want to compute a joint log-likelihood for action, boundary, and antecedent,
            # And we will use this joint LL for inference with beam search
            denom = util.logsumexp(torch.cat([
                action_logits,
                denom
            ], dim=-1), dim=-1)
            numer = util.logsumexp(torch.cat([
                action_logits + torch.where(
                    util.one_hot_ignore_negative(
                        labels, num_classes=3), 0., float("-inf")
                )[..., :2],
                numer
            ], dim=-1), dim=-1)

            loss = (denom - numer)[decoder_attention_mask.bool()].sum()
            loss = loss / target_ids.size(0)

            l_choice, typing_choice = None, None

        else:  # inference, step-by-step classifications
            # (batch_size, 1, dim)
            outputs.last_hidden_state = outputs.last_hidden_state.to(self.device)
            # denom: (batch_size, 1, 1)
            # l_choice / typing_choice: (batch_size, 1)
            (denom, l_choice, typing_choice) = self.get_logits_inference(
                outputs.last_hidden_state,  # current_hidden_state
                full_decoder_input_ids,     # previous decoded ids
                full_hidden_states=full_hidden_states
            )
            action_logits = self.action_head(outputs.last_hidden_state)
            # (batch_size, 1, 1)
            action_logits = torch.cat(
                [torch.zeros_like(action_logits),
                 action_logits,
                 denom],
                dim=-1
            )
            # Restore lm_logits from action_logits
            lm_logits = self.decoder_input_ids_to_vocab_mask(
                action_logits,
                full_decoder_input_ids,
                encoder_input_ids
            )

        return MySeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            pairing=l_choice,
            typing=typing_choice
        )

    def get_logits_inference(
        self,
        current_hidden_state,  # inference: (batch_size, 1, dim)
        decoder_input_ids,  # (batch_size, seq_len, )
        full_hidden_states=None,  # tuple((batch_size, 1, dim))
    ):
        # Shape: (batch_size, seq_len, )
        batch_size = decoder_input_ids.size(0)
        output_ids = decoder_input_ids[:, 1:]  # excluding decoder BOS

        range_vec = torch.arange(
            output_ids.size(1)-1, -1, -1,
            dtype=torch.long, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)

        if len(full_hidden_states) == 0:
            # the first valid token in the output
            return (
                current_hidden_state.new_full((batch_size, 1, self.num_typing_classes), float("-inf")), 
                decoder_input_ids.new_full((batch_size, 1), -1),
                decoder_input_ids.new_full((batch_size, 1), -1)
            )

        # concatenating into sequence: (batch_size, seq_len, dim)
        decoder_output = torch.cat(full_hidden_states, dim=1)

        # check special tokens
        # Shape: (batch_size, seq_len, )
        is_l = (output_ids == self.mention_start_id)

        if is_l.sum() == 0:
            # no full mention and no previous mentions
            lr_denom = current_hidden_state.new_full((batch_size, 1, self.num_typing_classes), float("-inf"))
            l_choice = decoder_input_ids.new_full((batch_size, 1), -1)
            typing_choice = decoder_input_ids.new_full((batch_size, 1), -1)
            denom = lr_denom
        else:
            # (batch_size, num_l, dim), (batch_size, num_l, )
            l_emb, l_emb_mask = batched_masked_select(decoder_output, is_l)
            
            # (batch_size, 1, num_l, )
            distance_to_previous_l, _ = batched_masked_select(range_vec, is_l)
            distance_to_previous_l = distance_to_previous_l.unsqueeze(1)

            # (batch_size, 1, num_l, 2*dim+feature_dim)
            lr_pair_emb = util.prepare_pair_embeddings(
                l_emb, current_hidden_state)

            # (batch_size, 1, num_l, self.num_typing_classes)
            lr_score = self.lr_scorer(lr_pair_emb)
            
            num_l_each_instance = is_l.sum(dim=-1)

            for i in range(batch_size):
                lr_score[i, :, :num_l_each_instance[i]-self.max_nest_depth, :] = NEGINF
                lr_score[i, :, num_l_each_instance[i]:, :] = NEGINF
                
            # (batch_size, 1)
            lr_denom = util.logsumexp(
                lr_score, dim=(2, 3)
            ).unsqueeze(-1)

            # (batch_size, 1, self.num_typing_classes)
            lr_score_max_over_entities, max_l = lr_score.max(dim=2)
            # (batch_size, 1)
            typing_choice = lr_score_max_over_entities.argmax(dim=2)
            # (batch_size, 1)
            l_choice = max_l.squeeze(1).gather(1, typing_choice)
            denom = lr_denom

        return (denom, l_choice, typing_choice)


    def get_logits_training(
        self,
        decoder_output,  # (batch_size, seq_len, hidden_dim)
        target_ids,      # (batch_size, seq_len, )
        lr_pair_flag=None,
        labels=None
    ):
        batch_size, seq_len = decoder_output.size(0), decoder_output.size(1)
        target_mask = (target_ids != self.config.pad_token_id)
        # (batch_size, seq_len)
        linearized_indices = (labels >= 0).cumsum(dim=-1) - 1
        # (batch_size, seq_len)
        is_l = (target_ids == self.mention_start_id)
        is_r = (target_ids == self.mention_end_id)

        if is_r.sum() == 0:
            numer, denom = (torch.full_like(decoder_output[...,:1], NEGINF),
                            torch.full_like(decoder_output[...,:1], NEGINF))
            return numer, denom

        # (batch_size, num_r / num_l)
        (l_pos, l_pos_mask) = batched_masked_select(linearized_indices, is_l)

        # (batch_size, num_r / num_l, hidden_dim)
        l_emb, _ = batched_masked_select(decoder_output, is_l)
        
        # (batch_size, seq_len, num_r / num_l)
        distance_to_previous_l = linearized_indices.unsqueeze(2) - l_pos.unsqueeze(1)
        
        # (batch_size, seq_len, num_r / num_l)
        is_after_l = (distance_to_previous_l > 0)
        is_after_l = is_after_l & target_mask.unsqueeze(2) & l_pos_mask.unsqueeze(1)

        # check correctness for batch
        kept_l = min(self.max_nest_depth, l_emb.size(1))
        # (batch_size, seq_len, kept_l)
        _, prev_l_indices = (
            -distance_to_previous_l + (is_after_l * 10000)
        ).topk(kept_l, dim=2)

        # (batch_size, seq_len, kept_l, hidden_dim)
        kept_l_emb = util.batch_select(l_emb, prev_l_indices)
        # (batch_size, seq_len, kept_l)
        distance_to_previous_l = util._batched_index_select(
            distance_to_previous_l, prev_l_indices, dim=2
        )

        expanded_decoder_output = decoder_output.unsqueeze(2).expand(
            -1, -1, kept_l, -1
        )
        # shape(batch_size, seq_len, kept_l, 2*hidden_dim)
        lr_pair_emb = torch.cat([kept_l_emb, expanded_decoder_output], dim=-1)

        # shape(batch_size, seq_len, kept_l, num_typing_classes)
        kept_is_after_l = is_after_l.gather(dim=2, index=prev_l_indices)
        lr_score = self.lr_scorer(lr_pair_emb) + (~kept_is_after_l).unsqueeze(-1) * NEGINF

        # (batch_size, seq_len, 1)
        lr_denom = logsumexp(
            lr_score,
            dim=(2, 3), keepdim=False
        ).unsqueeze(-1) * is_after_l.any(dim=2, keepdim=True)
        # (batch_size, seq_len, num_l, num_typing_classes) ->
        #     (batch_size, seq_len, kept_l, num_typing_classes)
        kept_lr_pair_flag = util._batched_index_select(
            lr_pair_flag, prev_l_indices, dim=2
        )
        # (batch_size, seq_len, 1)
        lr_numer = logsumexp(
            lr_score + (~kept_lr_pair_flag) * NEGINF,
            dim=(2, 3), keepdim=False
        ).unsqueeze(-1) * is_after_l.any(dim=2, keepdim=True)

        numer, denom = lr_numer, lr_denom

        return numer, denom


    def decoder_input_ids_to_vocab_mask(
        self,
        action_logits,
        decoder_input_ids,
        input_ids
    ):
        # counting how many words have been copied
        is_copied = ((decoder_input_ids != self.mention_start_id) &\
                     (decoder_input_ids != self.mention_end_id))
        # (batch_size, 1)
        num_copied = is_copied.sum(dim=-1, keepdim=True) - 1
        num_copied = num_copied.clamp(max=input_ids.size(1) - 1)

        # compute pointer to input tokens
        lm_logits = action_logits.new_full(
            (action_logits.size(0), action_logits.size(1), self.config.vocab_size),
            float("-inf")
        )
        word_to_copy = input_ids.expand(num_copied.size(0), -1)  # repeating over beams
        word_to_copy = word_to_copy.gather(1, num_copied)

        lm_logits.scatter_(2, word_to_copy.unsqueeze(-1), action_logits[:, :, :1])
        lm_logits[:, :, self.mention_start_id] = action_logits[:, :, 1]
        lm_logits[:, :, self.mention_end_id] = action_logits[:, :, 2]

        return lm_logits

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_encoder_input_ids=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            cut_decoder_input_ids = decoder_input_ids[:, -1:]
        else:
            cut_decoder_input_ids = decoder_input_ids

        if "full_hidden_states" not in kwargs:
            kwargs["full_hidden_states"] = []  # initializing the list

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_input_ids": decoder_encoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": cut_decoder_input_ids,   # last decoder_input_ids
            "full_decoder_input_ids": decoder_input_ids,  # full_decoder_input_ids
            "full_hidden_states": kwargs["full_hidden_states"],
            "decoder_pairing": kwargs["decoder_pairing"],
            "decoder_typing": kwargs["decoder_typing"],
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # override
    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: MySeq2SeqLMOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        if "full_hidden_states" not in model_kwargs:
            model_kwargs["full_hidden_states"] = []

        model_kwargs["full_hidden_states"].append(
            outputs.decoder_hidden_states[-1].to(outputs.pairing.get_device())
        )
        model_kwargs["decoder_pairing"].append(outputs.pairing)
        model_kwargs["decoder_typing"].append(outputs.typing)

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=-1
                )
        return model_kwargs

    def get_encoder(self):
        return self.t5.get_encoder()

    def get_decoder(self):
        return self.t5.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def _resize_action_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.action_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.action_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens), device=self.device)
            new_bias = torch.cat([self.action_logits_bias, extra_bias], dim=1)
        self.register_buffer("action_logits_bias", new_bias)

    def get_output_embeddings(self):
        return None

    def get_action_head(self):
        return self.action_head

    def set_output_embeddings(self, new_embeddings):
        self.action_head = new_embeddings

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx)
                      for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
