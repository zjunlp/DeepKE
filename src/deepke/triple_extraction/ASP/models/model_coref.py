"""
    Sequence to sequence model wrapper for coreference resolution.
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
from .t5_coref import T5Coref

logger = logging.getLogger(__file__)


class CorefModel(torch.nn.Module):
    """
        Model wrapper for coreference resolution.
    """
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.max_seg_len = config["max_segment_len"]
        self.device = device # main device

        # Model
        self.dropout = nn.Dropout(p=config["dropout_rate"])

        if 'device_map' in config:
            self.config['device_map'] = {
                int(k): v for k, v in config['device_map'].items()
            }

        self.tz = T5Tokenizer.from_pretrained("t5-small")

        self.SPEAKER_START = '<speaker>'
        self.SPEAKER_END   = '</speaker>'
        self.MENTION_START = '<m>'
        self.MENTION_END   = '</m>'

        self.tz.add_tokens(self.SPEAKER_START)
        self.tz.add_tokens(self.SPEAKER_END)
        self.tz.add_tokens(self.MENTION_START)
        self.tz.add_tokens(self.MENTION_END)

        self.mention_start_id = self.tz.convert_tokens_to_ids(self.MENTION_START)
        self.mention_end_id   = self.tz.convert_tokens_to_ids(self.MENTION_END)

        self.model = T5Coref.from_pretrained(
            config["plm_pretrained_name_or_path"],
            asp_hidden_dim=config["hidden_size"],
            asp_dropout_rate=config["dropout_rate"],
            asp_init_std=config["init_std"],
            asp_feature_emb_size=config["feature_emb_size"],
            asp_activation=config["activation"],
            mention_start_id=self.mention_start_id,
            mention_end_id=self.mention_end_id
        )

        self.beam_size = config["beam_size"]
        self.model.resize_token_embeddings(self.tz.vocab_size + 4)


    def parallel_preparation_training(self, ):
        if torch.cuda.device_count() == 1:
            self.model = self.model.cuda()
            return
        # prepare the model for parallel training
        if (not self.model.t5.model_parallel or
                self.model.emb_rr_distance.weight.get_device() != self.device):
            logger.info(
                f"Moving model to {self.device} and parallelize for training"
            )
            if not self.model.t5.model_parallel:
                self.model.t5.parallelize(
                    device_map=self.config['device_map'] if 'device_map' in self.config else None)
            self.model.rr_scorer = self.model.rr_scorer.to(self.device)
            self.model.lr_scorer = self.model.lr_scorer.to(self.device)
            self.model.action_head = self.model.action_head.to(self.device)
            self.model.emb_rr_distance = self.model.emb_rr_distance.to(self.device)
            torch.cuda.empty_cache()
        return


    def parallel_preparation_inference(self, ):
        if torch.cuda.device_count() == 1:
            self.model = self.model.cuda()
            return
        # prepare the model for parallel inference
        if self.model.emb_rr_distance.weight.get_device() != self.device:
            logger.info(
                f"Moving model from {self.model.emb_rr_distance.weight.get_device()} to {self.device} for inference"
            )
            self.model.rr_scorer = self.model.rr_scorer.to(self.device)
            self.model.lr_scorer = self.model.lr_scorer.to(self.device)
            self.model.action_head = self.model.action_head.to(self.device)
            self.model.emb_rr_distance = self.model.emb_rr_distance.to(self.device)
            if not self.model.t5.model_parallel:
                self.model.t5.parallelize()
            torch.cuda.empty_cache()
        return


    def get_params(self, named=False):
        plm_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if 't5' in name:
                to_add = (name, param) if named else param
                plm_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return plm_based_param, task_param


    def forward(
        self,
        input_ids, input_mask,
        target_ids, target_mask,
        action_labels, lr_pair_flag, rr_pair_flag,
        is_training,
        **kwargs,
    ):
        if len(is_training.size()) == 1:
            is_training = is_training[0]

        if (is_training == 1):  # training
            self.parallel_preparation_training()

            flag_grad_ckpt = False
            # reduce the memory usage with gradient checkpointing
            if target_ids.size(1) > 1400 and\
                ("3b" in self.config['plm_pretrained_name_or_path'].lower() or
                 "-xl" in self.config['plm_pretrained_name_or_path'].lower() or
                 "-xxl" in self.config['plm_pretrained_name_or_path'].lower()):
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
                use_cache=(not flag_grad_ckpt)
            )
            if flag_grad_ckpt:
                self.model.gradient_checkpointing_disable()
                flag_grad_ckpt = False
            total_loss = seq2seq_output.loss

            return total_loss

        else:  # inference
            self.parallel_preparation_inference()

            # save the decoded actions
            decoder_pairing, decoder_linking = [], []
            model_output = self.model.generate(
                input_ids,
                early_stopping=True,
                max_length=4096,
                num_beams=self.beam_size,
                num_return_sequences=self.beam_size,
                no_repeat_ngram_size=0,
                encoder_no_repeat_ngram_size=0,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                **{
                    "decoder_encoder_input_ids": input_ids,
                    "decoder_pairing": decoder_pairing,
                    "decoder_linking": decoder_linking
                }
            )
            results = {
                "output_ids": [],
                "start_token": [],
                "end_token": [],
                "pairing": [],
                "linking": []
            }

            for i in range(input_ids.size(0)):
                # Separatly decoding each instance in the batch
                # removing </s>
                output_ids = model_output.sequences[i][1:]

                # check special tokens
                is_start_token = (output_ids == self.mention_start_id)
                is_end_token = (output_ids == self.mention_end_id)

                range_vec = torch.arange(0, output_ids.size(0), device=self.device)
                start_token_pos = range_vec[is_start_token]
                end_token_pos = range_vec[is_end_token]
                results["output_ids"].append(output_ids)
                results["start_token"].append(start_token_pos)
                results["end_token"].append(end_token_pos)

                results["pairing"].append([x[i] for x in decoder_pairing])
                results["linking"].append([x[i] for x in decoder_linking])

            return results


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


    def decoding(self, output, stored_info):
        output_ids, pairing_decisions, antecedent_decisions = (
            output["output_ids"].tolist(),
            output["pairing"],
            output["linking"]
        )
        subtoken_map = stored_info["subtoken_map"]
        # mapping from generated sequence index to original index
        mapping = self.get_mapping_to_input_sequence(output_ids)

        mentions, start_ind = [], []
        predicted_clusters, antecedent_indices = [], []

        mention_to_cluster_id = {}

        for i in range(len(output_ids)):
            if output_ids[i] == self.tz.pad_token_id:
                break
            if output_ids[i] == self.mention_start_id:
                start_ind.append(i)
            if output_ids[i] == self.mention_end_id:
                mention = (
                    subtoken_map[mapping[start_ind[pairing_decisions[i]]]],
                    subtoken_map[mapping[i]]
                )
                mentions.append(mention)
                ant = antecedent_decisions[i]
                if ant == -1:
                    predicted_clusters.append([mention])
                    mention_to_cluster_id[mention] = len(predicted_clusters) - 1
                else:
                    antecedent = mentions[ant]
                    predicted_clusters[
                        mention_to_cluster_id[antecedent]
                    ].append(mention)
                    mention_to_cluster_id[mention] = mention_to_cluster_id[antecedent]

        mention_to_predicted = {
            m: tuple(predicted_clusters[cluster_idx])\
            for m, cluster_idx in mention_to_cluster_id.items()
        }
        predicted_clusters = [tuple(c) for c in predicted_clusters]

        result_dict = {
            "predicted": predicted_clusters,
            "mention_to_predicted": mention_to_predicted,
            "predicted_mentions": mentions
        }
        return result_dict

    def extract_gold_clusters_from_gold_annotation(
        self, 
        stored_info
    ):
        output_ids = stored_info['target_sentence']
        cluster_category = stored_info['cluster_category']
        mention_indice = stored_info['mention_indice']

        if type(output_ids[0]) == str:
            output_ids = [self.tz.convert_tokens_to_ids(x) for x in output_ids]
        # mapping from generated sequence index to original index
        mapping = self.get_mapping_to_input_sequence(output_ids)
        subtoken_map = stored_info["subtoken_map"]

        # Get predicted clusters
        mention_to_cluster_id = {}
        gold_clusters, mentions = [], []

        for i, (category_idx, boundary_idx) in enumerate(zip(cluster_category, mention_indice)):
            if category_idx == -1:
                continue
            # Add mention to cluster
            mention = (
                int(subtoken_map[mapping[boundary_idx]]), 
                int(subtoken_map[mapping[i]])
            )
            mentions.append(mention)
            while category_idx >= len(gold_clusters):
                gold_clusters.append([])

            gold_clusters[category_idx].append(mention)
            mention_to_cluster_id[mention] = category_idx

        gold_clusters = [tuple(c) for c in gold_clusters if len(c) > 1]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}

        result_dict = {
            "gold": gold_clusters,
            "mention_to_gold": mention_to_gold,
            "gold_mentions": mentions
        }
        return result_dict
