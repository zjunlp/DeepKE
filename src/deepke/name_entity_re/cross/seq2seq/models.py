import os
from copy import deepcopy
import torch
from torch import nn
from .modeling_t5 import T5ForConditionalGeneration

class T5Prompt(nn.Module):
    def __init__(self, model_name_or_path, config, args):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
        self.config = self.t5.config
        self.args = args
        
        # model config related
        self.match_n_layer = self.config.num_decoder_layers
        self.match_n_head = self.config.num_heads
        self.n_embd = self.config.d_model
        self.match_n_embd = self.config.d_kv
        
        # prefix related
        self.prompt_len = args.prompt_len
        self.prompt_dim = args.prompt_dim
        self.prompt_inputs = torch.arange(self.prompt_len).long()    # [0, 1, ..., prompt_len-1]
        
        self.wte = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        
        self.wte_enc = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        
        self.wte_dec = nn.Embedding(self.prompt_len, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.prompt_dim),
            nn.Tanh(),
            nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        
        self.dropout = nn.Dropout(0.1)
                    
        if args.source_prefix_path is not None and 'concat' in args.prefix_fusion_way:
            self.source_prompt_inputs = torch.arange(self.prompt_len).long()    # [0, 1, ..., prompt_len-1]
            self.source_wte = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )
            
            self.source_wte_enc = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )
            
            self.source_wte_dec = nn.Embedding(self.prompt_len, self.n_embd)
            self.source_control_trans_dec = nn.Sequential(
                nn.Linear(self.n_embd, self.prompt_dim),
                nn.Tanh(),
                nn.Linear(self.prompt_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )
            
        if args.multi_source_path:
            # module
            self.prefix_project_layer = nn.Sequential(
                nn.Linear(self.n_embd, 500),
                nn.Tanh(),
                nn.Linear(500, self.n_embd)
            )
                    
        if 'lstm' in args.prefix_fusion_way:
            self.lstm = nn.LSTM(input_size=self.match_n_head * self.match_n_embd, 
                                hidden_size=self.match_n_head * self.match_n_embd, 
                                num_layers=2, 
                                batch_first=True,
                                bidirectional=True)
            
        if args.freeze_plm:
            for name, param in self.t5.named_parameters():
                if 'encoder' in name or 'decoder' in name and 'adapter' not in name:
                    param.requires_grad = False
                    
        if args.freeze_prefix:
            for name, param in self.named_parameters():
                if 'wte' in name or 'control_trans' in name:
                    param.requires_grad = False
       
    def get_prompt(self, bsz=None):
        input_tokens = self.prompt_inputs.unsqueeze(0).expand(bsz, -1).to(self.t5.device)
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (self.prompt_inputs.unsqueeze(0).expand(bsz, -1)).to(self.t5.device)
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, prompt_len
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result
    
    
    def get_source_prompt(self, bsz=None):
        input_tokens = self.source_prompt_inputs.unsqueeze(0).expand(bsz, -1).to(self.t5.device)
        temp_control = self.source_wte(input_tokens)
        past_key_values = self.source_control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.source_wte_dec(input_tokens)
        past_key_values_dec = self.source_control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (self.source_prompt_inputs.unsqueeze(0).expand(bsz, -1)).to(self.t5.device)
        temp_control_enc = self.source_wte_enc(input_tokens_enc)
        past_key_values_enc = self.source_control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, prompt_len
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result
    
    def convert_prefix(self, prefix_dict):
        '''trans dict prefix to 1-D tensor prefix'''
        trans_prefix = []
        for i, prefix in enumerate(prefix_dict): # 12 layer
            # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
            current_key_prefix = []
            for key in prefix.keys():
                # prev_key', 'prev_value', 'prev_key_padding_mask'
                current_key_prefix.append(torch.stack([
                            prefix[key]['prev_key'].squeeze(0).transpose(1, 0).reshape(-1, self.n_embd).mean(0),
                            prefix[key]['prev_value'].squeeze(0).transpose(1, 0).reshape(-1, self.n_embd).mean(0),
                        ], dim=0).mean(0))  #  768
            trans_prefix.append(torch.stack(current_key_prefix, dim=0).mean(0))
        trans_prefix = torch.stack(trans_prefix, dim=0).mean(0)
        return trans_prefix


    def normalize_multi_source(self, label_word, tokenizer, device):
        '''This function is to normalize multi source to aggregate them.'''
        assert self.args.multi_source_path is not None
        
        # load multi source prefix and label word
        source_paths = self.args.multi_source_path.split(',')
        self.source_prefixes = [torch.load(os.path.join(path, 'prefix.pt')) for path in source_paths]
                
        trans_source_prefixes = []  # n x 768
        for source_prefix in self.source_prefixes: # n source
            trans_source_prefixes.append(self.convert_prefix(source_prefix))
        self.trans_source_prefixes = torch.stack(trans_source_prefixes, dim=0)
        
        self.trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1)).unsqueeze(0)
            
        source_label_words = [torch.load(os.path.join(path, 'label_word.pt')) for path in source_paths]
        for i in range(len(source_label_words)):
            source_label_words[i] = torch.stack(list(source_label_words[i].values()), dim=0).mean(0)
        self.source_label_words = torch.stack(source_label_words, dim=0).to(device)
        self.label_word_id = [tokenizer.encode(label, add_special_tokens=False) for label in label_word]
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        return_dict=True,
    ):
        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz=bsz)
        
        if self.args.multi_source_path:
            # 1. compute cosine similarity between target prefix and source prefixes
            trans_source_prefixes = self.prefix_project_layer(self.trans_source_prefixes.detach().to(input_ids.device))   # project
            
            trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1))
            trans_target_prefix = self.prefix_project_layer(trans_target_prefix).detach().unsqueeze(0)
            
            prefix_sim = torch.cosine_similarity(trans_target_prefix.to(input_ids.device), trans_source_prefixes.to(input_ids.device))   # n
            prefix_sim = prefix_sim / prefix_sim.sum()
            
            # 2. compute cosine similarity between target label word and source label words
            target_label_word = []
            for label_id in self.label_word_id:
                if len(label_id) > 1:
                    target_label_word.append(torch.stack([self.t5.shared.weight.data[id] for id in label_id], dim=0).mean(0))
                else:
                    target_label_word.append(self.t5.shared.weight.data[label_id[0]])
            target_label_word = torch.stack(target_label_word, dim=0).mean(0).to(input_ids.device)
            label_word_sim = torch.cosine_similarity(target_label_word.unsqueeze(0), self.source_label_words.to(input_ids.device))
            label_word_sim = label_word_sim / label_word_sim.sum()
            
            total_sim = (prefix_sim + label_word_sim) / 2

            # aggregate
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            agg_prefix = torch.zeros((1, self.match_n_head, self.prompt_len, self.match_n_embd), device=input_ids.device)
                            for j in range(len(self.source_prefixes)):
                                agg_prefix += self.source_prefixes[j][i][key][sub_key].to(input_ids.device) * total_sim[j]
                            past_prompt[i][key][sub_key] = (past_prompt[i][key][sub_key] + agg_prefix) / 2

        if 'concat' in self.args.prefix_fusion_way:
            source_past_prompt = self.get_source_prompt(bsz=bsz)
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            past_prompt[i][key][sub_key] = torch.cat((past_prompt[i][key][sub_key], 
                                                                      source_past_prompt[i][key][sub_key]), dim=2)
                        elif len(prefix[key][sub_key].size()) == 2:
                            past_prompt[i][key][sub_key] = torch.cat((past_prompt[i][key][sub_key], 
                                                                      source_past_prompt[i][key][sub_key]), dim=1)
        
        if 'lstm' in self.args.prefix_fusion_way:
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            _, _, prefix_len, _ = prefix[key][sub_key].size()
                            lstm_output = self.lstm(past_prompt[i][key][sub_key].reshape(bsz, prefix_len, self.match_n_head * self.match_n_embd))[0]
                            if lstm_output.size(-1) == 2 * (self.match_n_head * self.match_n_embd):   # bidirectional
                                past_prompt[i][key][sub_key] = lstm_output.reshape(bsz, self.match_n_head, prefix_len, 2, self.match_n_embd).mean(dim=-2)
                            else:
                                past_prompt[i][key][sub_key] = lstm_output.reshape(bsz, self.match_n_head, prefix_len, self.match_n_embd)
      

        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=return_dict,
            past_prompt=past_prompt,
        )
    
    
    def generate(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs,
    ):
        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz=bsz)
        
        if self.args.multi_source_path:
            # 1. compute cosine similarity between target prefix and source prefixes
            trans_source_prefixes = self.prefix_project_layer(self.trans_source_prefixes.detach().to(input_ids.device))   # project
            
            trans_target_prefix = self.convert_prefix(self.get_prompt(bsz=1))
            trans_target_prefix = self.prefix_project_layer(trans_target_prefix).detach().unsqueeze(0)
            
            prefix_sim = torch.cosine_similarity(trans_target_prefix.to(input_ids.device), trans_source_prefixes.to(input_ids.device))   # n
            prefix_sim = prefix_sim / prefix_sim.sum()
            
            # 2. compute cosine similarity between target label word and source label words
            target_label_word = []
            for label_id in self.label_word_id:
                if len(label_id) > 1:
                    target_label_word.append(torch.stack([self.t5.shared.weight.data[id] for id in label_id], dim=0).mean(0))
                else:
                    target_label_word.append(self.t5.shared.weight.data[label_id[0]])
            target_label_word = torch.stack(target_label_word, dim=0).mean(0).to(input_ids.device)
            label_word_sim = torch.cosine_similarity(target_label_word.unsqueeze(0), self.source_label_words.to(input_ids.device))
            label_word_sim = label_word_sim / label_word_sim.sum()
            
            total_sim = (prefix_sim + label_word_sim) / 2

            # aggregate
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            agg_prefix = torch.zeros((1, self.match_n_head, self.prompt_len, self.match_n_embd), device=input_ids.device)
                            for j in range(len(self.source_prefixes)):
                                agg_prefix += self.source_prefixes[j][i][key][sub_key].to(input_ids.device) * total_sim[j]
                            past_prompt[i][key][sub_key] = (past_prompt[i][key][sub_key] + agg_prefix) / 2
          
          
        if 'concat' in self.args.prefix_fusion_way:
            source_past_prompt = self.get_source_prompt(bsz=bsz)
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            past_prompt[i][key][sub_key] = torch.cat((past_prompt[i][key][sub_key], 
                                                                      source_past_prompt[i][key][sub_key]), dim=2)
                        elif len(prefix[key][sub_key].size()) == 2:
                            past_prompt[i][key][sub_key] = torch.cat((past_prompt[i][key][sub_key], 
                                                                      source_past_prompt[i][key][sub_key]), dim=1)
        
        if 'lstm' in self.args.prefix_fusion_way:
            for i, prefix in enumerate(past_prompt):    # 24
                for key in prefix.keys():               # 'decoder_prompt', 'cross_attention_prompt', 'encoder_prompt'
                    for sub_key in prefix[key].keys():  # prev_key', 'prev_value', 'prev_key_padding_mask'
                        if len(prefix[key][sub_key].size()) == 4:
                            _, _, prefix_len, _ = prefix[key][sub_key].size()
                            lstm_output = self.lstm(past_prompt[i][key][sub_key].reshape(bsz, prefix_len, self.match_n_head * self.match_n_embd))[0]
                            if lstm_output.size(-1) == 2 * (self.match_n_head * self.match_n_embd):   # bidirectional
                                past_prompt[i][key][sub_key] = lstm_output.reshape(bsz, self.match_n_head, prefix_len, 2, self.match_n_embd).mean(dim=-2)
                            else:
                                past_prompt[i][key][sub_key] = lstm_output.reshape(bsz, self.match_n_head, prefix_len, self.match_n_embd)
      
        
        generated_ids = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids