"""
    Sequence to sequence model wrapper for End-to-End Relation Extraction
    Decoding algorithms, parallelism handling, and other utilities
    Tianyu Liu
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
import logging

from transformers import T5Tokenizer
from .t5_ere import T5ERE

logger = logging.getLogger(__file__)


class EREModel(torch.nn.Module):
    """
        Model wrapper for joint entity and relation extraction.
    """
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.max_seg_len = config["max_segment_len"]
        self.device = device

        # Model
        self.dropout = nn.Dropout(p=config["dropout_rate"])
        
        if 'device_map' in config:
            self.config['device_map'] = {
                int(k):v for k, v in config['device_map'].items()
            }

        self.tz = T5Tokenizer.from_pretrained("t5-small")
        self.MENTION_START = '<m>'
        self.MENTION_END   = '</m>'
        self.tz.add_tokens(self.MENTION_START)
        self.tz.add_tokens(self.MENTION_END)

        self.mention_start_id = self.tz.convert_tokens_to_ids(self.MENTION_START)
        self.mention_end_id   = self.tz.convert_tokens_to_ids(self.MENTION_END)

        self.model = T5ERE.from_pretrained(
            config['plm_pretrained_name_or_path'],
            asp_hidden_dim=config["hidden_size"],
            asp_dropout_rate=config["dropout_rate"],
            asp_init_std=config["init_std"],
            asp_feature_emb_size=config["feature_emb_size"],
            asp_activation=config["activation"],
            num_typing_classes=config["num_typing_classes"],
            num_linking_classes=config["num_linking_classes"],
            mention_start_id=self.mention_start_id,
            mention_end_id=self.mention_end_id
        )

        self.beam_size = config["beam_size"]
        self.model.resize_token_embeddings(
            self.tz.vocab_size + 2
        )


    def parallel_preparation_training(self, ):
        if torch.cuda.device_count() == 1:
            self.model = self.model.cuda()
            return
        # prepare the model for parallel training
        if (not self.model.t5.model_parallel or
                self.model.action_head[0].weight.get_device() != self.device):
            logger.info(
                f"Moving model to {self.device} and parallelize for training"
            )
            if not self.model.t5.model_parallel:
                self.model.t5.parallelize(
                    device_map=self.config['device_map'] if 'device_map' in self.config else None)
            self.model.rr_scorer = self.model.rr_scorer.to(self.device)
            if hasattr(self.model, 'rr_scorer_rev'):
                self.model.rr_scorer_rev = self.model.rr_scorer_rev.to(self.device)
            self.model.lr_scorer = self.model.lr_scorer.to(self.device)
            self.model.action_head = self.model.action_head.to(self.device)

            if hasattr(self.model, 'emb_l_distance'):
                self.model.emb_l_distance = self.model.emb_l_distance.to(self.device)
            if hasattr(self.model, 'emb_r_distance'):
                self.model.emb_r_distance = self.model.emb_r_distance.to(self.device)
            torch.cuda.empty_cache()
        return

    def parallel_preparation_inference(self, ):
        if torch.cuda.device_count() == 1:
            self.model = self.model.cuda()
            return
        # prepare the model for parallel inference
        if self.model.action_head[0].weight.get_device() != self.device:
            logger.info(
                f"Moving model from {self.model.action_head[0].weight.get_device()} to {self.device} for inference"
            )
            self.model.rr_scorer = self.model.rr_scorer.to(
                self.device)
            if hasattr(self.model, 'rr_scorer_rev'):
                self.model.rr_scorer_rev = self.model.rr_scorer_rev.to(self.device)
            self.model.lr_scorer = self.model.lr_scorer.to(self.device)
            self.model.action_head = self.model.action_head.to(self.device)

            if hasattr(self.model, 'emb_l_distance'):
                self.model.emb_l_distance = self.model.emb_l_distance.to(self.device)
            if hasattr(self.model, 'emb_r_distance'):
                self.model.emb_r_distance = self.model.emb_r_distance.to(self.device)
            if not self.model.t5.model_parallel:
                self.model.t5.parallelize()
            torch.cuda.empty_cache()
        return

    def get_params(self, named=False):
        plm_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if 't5' in name: #  or 'action_head' in name:
                to_add = (name, param) if named else param
                plm_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)

        return plm_based_param, task_param

    def forward(
            self,
            input_ids, input_mask, to_copy_ids,
            target_ids, target_mask, action_labels,
            lr_pair_flag, rr_pair_flag,
            is_training,
            sentence_idx=None,
            target_sentence_idx=None,
            same_sentence_flag=None,
            **kwargs, 
        ):
        if len(is_training.size()) == 1:
            is_training = is_training[0]
        
        if (is_training == 1): # training
            self.parallel_preparation_training()
            
            flag_grad_ckpt = False
            if target_ids.size(1) > 1024:
                self.model.gradient_checkpointing_enable()
                flag_grad_ckpt = True

            seq2seq_output = self.model(
                input_ids=input_ids, 
                attention_mask=input_mask, 
                decoder_input_ids=target_ids,
                decoder_attention_mask=target_mask,
                labels=action_labels,
                output_hidden_states=True,
                lr_pair_flag=lr_pair_flag,
                rr_pair_flag=rr_pair_flag,
                same_sentence_flag=same_sentence_flag,
                target_sentence_idx=target_sentence_idx,
                use_cache=(not flag_grad_ckpt)
            )
            if flag_grad_ckpt:
                self.model.gradient_checkpointing_disable()
                flag_grad_ckpt = False
            total_loss = seq2seq_output.loss
            
            return total_loss
        
        else: # inference
            self.parallel_preparation_inference()

            # save the decoded actions
            decoder_pairing, decoder_linking, decoder_typing = [], [], []
            model_output = self.model.generate(
                input_ids, 
                early_stopping=True,
                max_length=196,
                num_beams=self.beam_size,
                num_return_sequences=self.beam_size,
                no_repeat_ngram_size=0,
                encoder_no_repeat_ngram_size=0,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                **{
                    "decoder_encoder_input_ids": to_copy_ids,
                    "decoder_pairing": decoder_pairing,
                    "decoder_linking": decoder_linking,
                    "decoder_typing": decoder_typing
                }
            )

            results = {
                "output_ids": [],
                "start_token": [],
                "end_token": [],
                "pairing": [],
                "linking": [],
                "typing": []
            }
            # taking the best sequence in the beam, removing </s>
            for i in range(input_ids.size(0)):
                output_ids = model_output.sequences[i][1:]

                # check special tokens
                is_start_token = (output_ids == self.mention_start_id)
                is_end_token = (output_ids == self.mention_end_id)

                range_vec = torch.arange(0, output_ids.size(0), device=self.device)
                start_token_pos = range_vec[is_start_token]
                end_token_pos = range_vec[is_end_token]
                results["start_token"].append(start_token_pos)
                results["end_token"].append(end_token_pos)
                results["output_ids"].append(output_ids)

                results["pairing"].append([x[i] for x in decoder_pairing])
                results["linking"].append([x[i] for x in decoder_linking])
                results["typing"].append([x[i] for x in decoder_typing])

            return results
            

    def extract_gold_res_from_gold_annotation(
        self,
        tensor_example,
        stored_info=None
    ):
        output_ids = tensor_example["target_ids"]
        mapping = self.get_mapping_to_input_sequence(output_ids)

        ent_types, rel_types, ent_indices, rel_indices = (
            tensor_example["ent_types"], tensor_example["rel_types"],
            tensor_example["ent_indices"], tensor_example["rel_indices"]
        )
        subtoken_map = stored_info["subtoken_map"]

        entities, relations, start_ind = [], [], []
        entity_map = {}
        # reconstructing mention_indices and antecedent_indices
        for i in range(len(output_ids)):
            if output_ids[i] == self.mention_start_id:
                start_ind.append(i)
            if output_ids[i] == self.mention_end_id:
                entity = (
                    int(subtoken_map[int(mapping[start_ind[-1]])]), # no nested
                    int(subtoken_map[int(mapping[i])]),
                    int(ent_types[i])
                )
                entities.append(entity)

                entity_map[i] = entity
                this_rel_types = rel_types[i].tolist()
                this_rel_ants = rel_indices[i].tolist()

                for _, (ant_tail, rel_type) in enumerate(
                        zip(this_rel_ants, this_rel_types)):
                    if rel_type != -1:
                        relations.append(
                            (entity_map[ant_tail][:2], entity[:2], int(rel_type),
                             entity_map[ant_tail][2], entity[2])
                        )
        result_dict = {
            "gold_entities": entities,
            "gold_relations": relations
        }
        return result_dict
        

    def decoding(self, output, stored_info):
        output_ids, pairing_decisions, typing_decisions, antecedent_decisions = (
            output["output_ids"].tolist(),
            output["pairing"],
            output["typing"],
            output["linking"]
        )
        subtoken_map = stored_info["subtoken_map"]
        sentence_idx = stored_info['sentence_idx'] if "sentence_idx" in stored_info else None

        mapping = self.get_mapping_to_input_sequence(output_ids)
        entities, relations, start_ind = [], [], []
        # reconstructing mention_indices and antecedent_indices
        for i in range(len(output_ids)):
            if output_ids[i] == self.tz.pad_token_id:
                break
            if output_ids[i] == self.mention_start_id:
                start_ind.append(i)
            if output_ids[i] == self.mention_end_id:
                this_type = int(typing_decisions[i])
                entity = (
                    subtoken_map[mapping[start_ind[pairing_decisions[i]]]],
                    subtoken_map[mapping[i]],
                    this_type,
                    sentence_idx[mapping[i]] if sentence_idx is not None else 0
                )
                entities.append(entity)

                this_rel_types = antecedent_decisions[i].tolist()
                for ent_id, rel_type in enumerate(this_rel_types):
                    if rel_type != -1:
                        relations.append(
                            (entities[ent_id][:2], entity[:2], int(rel_type),
                             entities[ent_id][2], entity[2])
                        )
                        if entities[ent_id][-1] != entity[-1]:
                            relations = relations[:-1]

        result_dict = {
            "predicted_entities": [x[:3] for x in entities],
            "predicted_relations": [x for x in relations],
        }
        return result_dict

    def get_mapping_to_input_sequence(
        self, output_ids
    ):
        # Get the mapping from the output with special tokens
        # to the input without special tokens.
        mapping, new_id = [], -1
        for i in range(len(output_ids)):
            if output_ids[i] == self.mention_start_id:
                new_id += 1
            elif output_ids[i] == self.mention_end_id:
                new_id += 0
            else:
                new_id += 1
            mapping.append(new_id)
            if output_ids[i] == self.mention_start_id:
                new_id -= 1

        return mapping
