import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_bart import BartConfig
from .modeling_bart import BartModel, _prepare_bart_decoder_inputs

from ..utils import avg_token_embeddings, seq_to_mask,get_model_device
from functools import partial
from typing import Union


class PromptBartEncoder(nn.Module):
    def __init__(self, encoder):
        super(PromptBartEncoder, self).__init__()
        self.bart_encoder = encoder
    
    def forward(self, src_tokens, attention_mask=None, past_key_values=None):
        encoder_dicts = self.bart_encoder(input_ids=src_tokens, attention_mask=attention_mask, past_key_values=past_key_values, return_dict=True, output_hidden_states=True)
        return encoder_dicts.last_hidden_state, encoder_dicts.hidden_states
        
class PromptBartDecoder(nn.Module):
    def __init__(self, decoder, pad_token_id, label_ids, use_prompt=False, prompt_len=10, learn_weights=False):
        super(PromptBartDecoder, self).__init__()
        self.bart_decoder = decoder
        self.pad_token_id = pad_token_id
        self.use_prompt = use_prompt
        self.prompt_len = prompt_len
        self.learn_weights = learn_weights
        self.label_ids = label_ids

        print(label_ids)
        if self.learn_weights:   # set learnable averge weights
            self.averge_weights = nn.ParameterList(parameters=None)
            for id in label_ids:
                if len(id) > 1:
                    self.averge_weights.append(nn.Parameter(torch.FloatTensor(len(id)).uniform_(1.0, 2.5)))
            print(self.averge_weights)
            mapping = [0, 2]
            for id in label_ids:
                mapping += id[:1]
            mapping = torch.LongTensor(mapping)
        else:
            mapping = torch.LongTensor([0, 2]+label_ids)
            self.label_start_id = min(label_ids)
            self.label_end_id = max(label_ids)+1

        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)

        hidden_size = decoder.embed_tokens.weight.size(1)
        self.bart_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(0.3),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
        self.dropout_layer = nn.Dropout(0.3)
    
    def forward(self, tgt_tokens, prompt_state):
        cumsum = tgt_tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[-1]).ne(cumsum[:, -1:])

        encoder_outputs = prompt_state.encoder_output  # last_hidden_state
        attention_mask = prompt_state.encoder_mask   # attention_mask
        first = prompt_state.first
        src_tokens = prompt_state.src_tokens
        past_key_values = prompt_state.past_key_values

        # mapping target tokens
        mapping_token_mask = tgt_tokens.lt(self.src_start_index) 
        mapped_tokens = tgt_tokens.masked_fill(tgt_tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tgt_tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)

        decoder_input_ids, _, causal_mask = _prepare_bart_decoder_inputs(
                self.pad_token_id, 
                tokens,
                decoder_input_ids=None,
                decoder_padding_mask=None,
                causal_mask_dtype=self.bart_decoder.embed_tokens.weight.dtype
        )

        if self.use_prompt:
            assert past_key_values is not None
            _, _, seqlen, _ = past_key_values[0]['self']['prev_value'].shape
            tgt_len = decoder_input_ids.size(1)
            temp_mask = torch.zeros(tgt_len, seqlen).to(causal_mask.device) #tgtlen, preseqlen
            causal_mask = torch.cat([temp_mask, causal_mask],dim=1) #tgtlen, preseqlen+tgtlen

        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id) 
            dict = self.bart_decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,  # last_hidden_state
                                encoder_padding_mask=attention_mask,  # attention_mask
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=causal_mask[:tokens.size(1), :self.prompt_len+tokens.size(1)],
                                output_hidden_states=True,
                                past_key_values=past_key_values,
                                return_dict=True)
        else:
            past_key_values = prompt_state.past_key_values
            dict = self.bart_decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=attention_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True)
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            prompt_state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        # compute eos scores
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[2:3]))  # bsz x max_len x 1
        
        if self.learn_weights:   # use averge_weights compute entity labels scores
            tag_scores = None
            idx = 0
            for ids in self.label_ids: # bsz x max_len x num_class
                if len(ids) <= 1:
                    temp_score = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[ids]))
                else:
                    weight = F.softmax(self.averge_weights[idx])
                    temp_score = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[[ids[0]]])) * weight[0]
                    for i in range(1, len(ids)):
                        temp_score = temp_score + F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[[ids[i]]])) * weight[i]
                    idx += 1
                if tag_scores is None:
                    tag_scores = temp_score
                else:
                    tag_scores = torch.cat((tag_scores, temp_score), dim=2)
        else:
            tag_scores = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))  # bsz x max_len x num_class

        # bsz x max_bpe_len x hidden_size
        src_outputs = encoder_outputs
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = attention_mask.eq(0)
            # src_outputs = self.decoder.embed_tokens(src_tokens)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.bart_decoder.embed_tokens(src_tokens))  # bsz x max_word_len x hidden_size
        src_outputs = (src_outputs + input_embed)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores
        return logits, prompt_state

    def decode(self, tokens, state):
        return self(tokens, state)[0][:, -1]

class PromptBartModel(nn.Module):
    def __init__(self, tokenizer, label_ids, args):
        super(PromptBartModel, self).__init__()
        self.use_prompt = args.use_prompt
        self.prompt_len = args.prompt_len
        self.prompt_dim = args.prompt_dim
        self.learn_weights = args.learn_weights
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        bart_name = args.bart_name

        self.bart_config = BartConfig.from_pretrained(bart_name)
        self.bart_config.use_prompt = args.use_prompt
        self.bart_config.preseqlen = args.prompt_len
        bart_config = self.bart_config
        bart_model = BartModel.from_pretrained(bart_name, config=bart_config)
        num_tokens, _ = bart_model.encoder.embed_tokens.weight.shape
        bart_model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        bart_model = avg_token_embeddings(tokenizer, bart_model, bart_name, num_tokens)
        
        self.prompt_encoder = PromptBartEncoder(bart_model.encoder)
        self.prompt_decoder = PromptBartDecoder(bart_model.decoder, tokenizer.pad_token_id, label_ids, self.use_prompt, self.prompt_len, self.learn_weights)

        self.prompt_inputs = torch.arange(self.prompt_len).long()
        self.encoder_prompt_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.encoder_mlp = nn.Sequential(
                            nn.Linear(bart_config.d_model, self.prompt_dim),
                            nn.Tanh(),
                            nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        
        self.decoder_prompt_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.decoder_mlp = nn.Sequential(
                            nn.Linear(bart_config.d_model, self.prompt_dim),
                            nn.Tanh(),
                            nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        
        self.prompt_cross_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.cross_mlp = nn.Sequential(
                            nn.Linear(bart_config.d_model, self.prompt_dim),
                            nn.Tanh(),
                            nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        
        self.dropout = nn.Dropout(0.0)


    def forward(self, src_tokens, tgt_tokens, src_seq_len, first):
        prompt_state = self.generator(src_tokens, src_seq_len, first)
        
        decoder_outputs, prompt_state =  self.prompt_decoder(tgt_tokens, prompt_state)
        return decoder_outputs


    def generator(self, src_tokens, src_seq_len, first):
        batch_size = src_tokens.size(0)
        past_key_values = self.get_prompt(batch_size) if self.use_prompt else None
        attention_mask = seq_to_mask(src_seq_len, max_len=src_tokens.size(1))
        encoder_outputs, hidden_states = self.prompt_encoder(src_tokens, attention_mask=attention_mask, past_key_values=past_key_values)
        prompt_state = PromptBartState(encoder_outputs, attention_mask, past_key_values, src_tokens, first, hidden_states[0], self.bart_config.preseqlen)

        return prompt_state

 
    def get_prompt(self, batch_size):
        input_tokens = self.prompt_inputs.unsqueeze(0).expand(batch_size, -1).to(self.device)

        # encoder prompt
        encoder_embed = self.encoder_prompt_embed(input_tokens)
        past_key_values = self.encoder_mlp(encoder_embed) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.bart_config.decoder_layers * 2, 
                                                self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # key + value

        # decoder prompt
        decoder_embed = self.decoder_prompt_embed(input_tokens)
        past_key_values2 = self.decoder_mlp(decoder_embed)  # bsz, seqlen, layer*emb
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.bart_config.decoder_layers * 2, 
                                                 self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)

        # cross prompt
        cross_embed = self.prompt_cross_embed(input_tokens)
        past_key_values_enc = self.cross_mlp(cross_embed)  # bsz, seqlen, layer*emb
        past_key_values_enc = past_key_values_enc.view(bsz, seqlen, self.bart_config.decoder_layers * 2, 
                                                       self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                    "prev_value": key_val[1].contiguous(),
                                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool() #bsz, preseqlen
                                    },
                        }
            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                            "prev_value": key_val2[1].contiguous(),
                                            "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val2.device).bool()
                                            }
            key_val_enc = past_key_values_enc[i]
            temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                    "prev_value": key_val_enc[1].contiguous(),
                                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val_enc.device).bool()
                                    }
            result.append(temp_dict)

        return result

    
class PromptBartState(object):
    def __init__(self, encoder_output, encoder_mask, past_key_values, src_tokens, first, src_embed_outputs, preseqlen):
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self.past_key_values = past_key_values
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.preseqlen = preseqlen

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new

    def num_samples(self):
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

class PromptGeneratorModel(nn.Module):
    def __init__(self, prompt_model, max_length=20, max_len_a=0.0, num_beams=1,
                 do_sample=False, bos_token_id=None, eos_token_id=None,
                 repetition_penalty=1, length_penalty=1.0, pad_token_id=0, restricter=None):
        super(PromptGeneratorModel, self).__init__()
        self.prompt_model = prompt_model
        self.decoder = prompt_model.prompt_decoder

        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a,
                                     num_beams=num_beams,
                                     bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                     repetition_penalty=repetition_penalty,
                                     length_penalty=length_penalty, pad_token_id=pad_token_id,
                                     restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a


    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, first=None):
        """
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.prompt_model(src_tokens, tgt_tokens, src_seq_len, first)


    def predict(self, src_tokens, src_seq_len=None, first=None):
        """
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        prompt_state = self.prompt_model.generator(src_tokens, src_seq_len, first)    # encoder output
        result = self.generate_func(tokens=None, state=prompt_state)
        return result


@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=1,
                    bos_token_id=None, eos_token_id=None, pad_token_id=0,
                    repetition_penalty=1, length_penalty=1.0, restricter=None):
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                             bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                             repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                             pad_token_id=pad_token_id, restricter=restricter)
    else:
        token_ids = _beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a,
                                          num_beams=num_beams,
                                          bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False,
                                          repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                                          pad_token_id=pad_token_id, restricter=restricter)

    return token_ids


def _no_beam_search_generate(decoder: PromptBartDecoder, state, tokens=None, max_length=20, max_len_a=0.0, bos_token_id=None,
                             eos_token_id=None,
                             repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0,
                             restricter=None):
    device = get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples()
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples() == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    scores = decoder.decode(tokens=tokens, state=state)  # update state
    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))
    # tokens = tokens[:, -1:]

    if max_len_a!=0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0), ), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)

    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids, state=state)  # batch_size x vocab_size

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores) 

        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)

        if _eos_token_id!=-1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len+1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id) 
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break
    return token_ids


def _beam_search_generate(decoder: PromptBartDecoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=4,
                          bos_token_id=None, eos_token_id=None, do_sample=True,
                          repetition_penalty=1.0, length_penalty=None, pad_token_id=0,
                          restricter=None) -> torch.LongTensor:
    assert do_sample is False
    # beam search
    device = get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        batch_size = state.num_samples
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `state`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples == batch_size, "The number of samples in `tokens` and `state` should match."

    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id

    scores = decoder.decode(tokens=tokens, state=state)
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

    scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
    if restricter is not None:
        _next_scores, _next_tokens = restricter(state, tokens, scores, num_beams+1)
    else:
        # bsz x (num_beams+1)
        _next_scores, _next_tokens = torch.topk(scores, num_beams+1, dim=1, largest=True, sorted=True)

    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    state.reorder_state(indices)
    tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length

    if max_len_a!=0:
        # (bsz x num_beams, )
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float()*max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams, ), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long()*max_length
        else:
            max_lengths = tokens.new_full((batch_size*num_beams,), fill_value=max_length, dtype=torch.long)
    hypos = [
        BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]

    not_eos_mask = _next_tokens.ne(_eos_token_id)  
    keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  
    keep_mask = not_eos_mask.__and__(keep_mask) 

    next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams) 
    next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)

    rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)

    if len(rows)>0: 
        for row, col in zip(rows.tolist(), cols.tolist()):
            _token = torch.cat([tokens[row*num_beams], _next_tokens[row, col:col+1]], dim=0)
            hypos[row].add(_token.clone(), _next_scores[row, col].item())

    # (batch_size, cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    cur_len = token_ids.size(1)

    # 0, num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < real_max_length:
        scores = decoder.decode(token_ids, state)  # (bsz x num_beams, vocab_size)
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if _eos_token_id!=-1:
            max_len_eos_mask = max_lengths.eq(cur_len+1)
            eos_scores = scores[:, _eos_token_id]
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores+1e32, eos_scores)

        scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
        _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
        _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
 
        if restricter is not None:
            next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
        else:
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)  # (bsz, 2*num_beams)
        from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
        next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        not_eos_mask = next_tokens.ne(_eos_token_id)  
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  
        keep_mask = not_eos_mask.__and__(keep_mask) 

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams) 
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        flag = True
        if cur_len+1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size) 
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1) 
        else:
            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]
            else:
                flag = False

        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                     eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    if _eos_token_id!=-1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)

        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten
        state.reorder_state(reorder_inds)
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or \
                               max_lengths[batch_idx*num_beams]==cur_len+1

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new_zeros(batch_size)
    best = []

    for i, hypotheses in enumerate(hypos):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        if _eos_token_id!=-1:
            best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1)*_eos_token_id])
        tgt_len[i] = len(best_hyp)
        best.append(best_hyp)

    # generate target batch
    decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i]] = hypo

    return decoded


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
