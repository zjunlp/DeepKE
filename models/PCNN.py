# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN(nn.Module):
    def __init__(self, opt):
        super(PCNN, self).__init__()

        self.opt = opt
        self.model_name = 'PCNN'
        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)
        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2

        # encoding sentence level feature via cnn
        self.convs = nn.ModuleList([
            nn.Conv2d(1,
                      self.opt.filters_num, (k, feature_dim),
                      padding=(int(k / 2), 0)) for k in self.opt.filters
        ])
        all_filter_num = self.opt.filters_num * len(self.opt.filters)
        self.cnn_linear = nn.Linear(all_filter_num, self.opt.sen_feature_dim)
        # self.cnn_linear = nn.Linear(all_filter_num, self.opt.rel_num)

        # concat the lexical feature in the out architecture
        self.out_linear = nn.Linear(all_filter_num + self.opt.word_dim * 6,
                                    self.opt.rel_num)
        # self.out_linear = nn.Linear(self.opt.sen_feature_dim, self.opt.rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)
        self.init_word_emb()
        self.init_model_weight()

    def init_model_weight(self):
        # use xavier to init
        nn.init.xavier_normal_(self.cnn_linear.weight)
        nn.init.constant_(self.cnn_linear.bias, 0.)
        nn.init.xavier_normal_(self.out_linear.weight)
        nn.init.constant_(self.out_linear.bias, 0.)
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def init_word_emb(self):

        w2v = torch.from_numpy(np.load(self.opt.w2v_path))

        # w2v = torch.div(w2v, w2v.norm(2, 1).unsqueeze(1))
        # w2v[w2v != w2v] = 0.0

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
        else:
            self.word_embs.weight.data.copy_(w2v)

    def forward(self, x):

        lexical_feature, word_feautre, left_pf, right_pf = x

        # No USE: lexical word embedding
        batch_size = lexical_feature.size(0)
        lexical_level_emb = self.word_embs(
            lexical_feature)  # (batch_size, 6, word_dim
        lexical_level_emb = lexical_level_emb.view(batch_size, -1)
        # lexical_level_emb = lexical_level_emb.sum(1)

        # sentence level feature
        word_emb = self.word_embs(
            word_feautre)  # (batch_size, max_len, word_dim)
        left_emb = self.pos1_embs(left_pf)  # (batch_size, max_len, word_dim)
        right_emb = self.pos2_embs(right_pf)  # (batch_size, max_len, word_dim)

        sentence_feature = torch.cat(
            [word_emb, left_emb, right_emb],
            2)  # (batch_size, max_len, word_dim + pos_dim *2)

        # conv part
        x = sentence_feature.unsqueeze(1)
        x = self.dropout(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        #  sen_level_emb = self.cnn_linear(x)
        #  sen_level_emb = self.tanh(sen_level_emb)
        sen_level_emb = x
        # combine lexical and sentence level emb
        x = torch.cat([lexical_level_emb, sen_level_emb], 1)
        x = self.dropout(x)
        x = self.out_linear(x)

        return x
