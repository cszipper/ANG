# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

"""
add entropy
based on dropout version
"""

class Encoder(nn.Module):
    """docstring for Encoder."""
    def __init__(self, args, rel_size, ent_size):
        super(Encoder, self).__init__()
        self.args = args
        self.pretraining = args.pretrain
        self.rel_size = rel_size
        self.ent_size = ent_size
        self.hidden_dim = args.latent_dim
        self.rel_emb = nn.Embedding(rel_size+3, self.hidden_dim)
        self.ent_emb = nn.Embedding(ent_size, self.hidden_dim)
        self.enc_gru = nn.GRU(self.hidden_dim, self.hidden_dim, bidirectional=True)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, input):
        src = self.ent_emb(input[0])
        rel = self.rel_emb(input[1])
        dst = self.ent_emb(input[2])
        EOT = self.rel_emb(input[3])
        embedded = torch.stack((src, rel, dst, EOT), dim=0).squeeze()

        output, hidden = self.enc_gru(embedded)

        return output, hidden

class Decoder(nn.Module):
    """docstring for Decoder."""
    def __init__(self, args, rel_size, ent_size):
        super(Decoder, self).__init__()
        self.args = args
        self.pretraining = args.pretrain
        self.teacher_forcing_ratio = 0.5
        self.rel_size = rel_size
        self.ent_size = ent_size
        self.hidden_dim = args.latent_dim * 2 # for bidirectional encoder
        self.rel_emb = nn.Embedding(rel_size+3, self.hidden_dim)
        self.ent_emb = nn.Embedding(ent_size, self.hidden_dim)
        self.dec_gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.rel_linear = nn.Linear(self.hidden_dim, rel_size+3)
        self.ent_linear = nn.Linear(self.hidden_dim, ent_size)
        self.output_mask = nn.Dropout(p=0.5)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, input, hidden, mask):
        SOT = F.relu(self.rel_emb(input[0]))
        src = F.relu(self.ent_emb(input[1]))
        rel = F.relu(self.rel_emb(input[2]))
        dst = F.relu(self.ent_emb(input[3]))

        input_list = [SOT, src, rel, dst]
        output_list = []
        hidden_list = []
        this_input = input_list[0].view(1, -1, self.hidden_dim)
        this_hidden = hidden
        for i in range(len(input_list)):
            output, hidden = self.dec_gru(this_input, this_hidden)
            output_list.append(self.output_mask(output))
            hidden_list.append(hidden)
            this_hidden = hidden_list[-1]
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            if self.training and i<len(input_list)-1 and use_teacher_forcing:
                this_input = input_list[i+1].view(1, -1, self.hidden_dim)
            else:
                this_input = output_list[-1]

        src_logit = self.ent_linear(output_list[0])
        rel_logit = self.rel_linear(output_list[1])
        dst_logit = self.ent_linear(output_list[2])
        EOT_logit = self.rel_linear(output_list[3])
        probs = None
        logits = None
        entropy = None
        if self.pretraining:
            _, src_gen = F.log_softmax(src_logit, dim=-1).topk(1)
            _, dst_gen = F.log_softmax(dst_logit, dim=-1).topk(1)
            _, rel_gen = F.log_softmax(rel_logit, dim=-1).topk(1)
            _, EOT_gen = F.log_softmax(EOT_logit, dim=-1).topk(1)
            output = (src_gen.squeeze(), rel_gen.squeeze(), dst_gen.squeeze(), EOT_gen.squeeze())
            logits = (src_logit, rel_logit, dst_logit, EOT_logit)
        else:
            if mask:
                sample = Categorical(logits=src_logit)
                gen = sample.sample()
                probs = sample.log_prob(gen)
                entropy = sample.entropy()
                output = (gen.view(-1, 1), input[2], input[3])
            else:
                sample = Categorical(logits=dst_logit)
                gen = sample.sample()
                probs = sample.log_prob(gen)
                entropy = sample.entropy()
                output = (input[1], input[2], gen.view(-1, 1))

        return output, hidden, logits, probs, entropy

class Seq2Seq(nn.Module):
    """docstring for Seq2Seq."""
    def __init__(self, args, rel_size, ent_size):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.pretraining = args.pretrain
        self.rel_size = rel_size
        self.ent_size = ent_size
        self.encoder = Encoder(args, rel_size, ent_size)
        self.decoder = Decoder(args, rel_size, ent_size)
        self.criterion = nn.CrossEntropyLoss()

    def loss_function(self, input, target):
        loss = 0
        for i in range(target.size(0)):
            loss += self.criterion(input[i].squeeze(), target[i].squeeze())
        return loss

    def policy_gradient(self, rewards, prob, entropy, alpha=0.01):
        # alpha = 0.01 by default
        loss = -torch.sum(prob * rewards + alpha * entropy)
        return loss

    def cat_hidden(self, encoder_hidden):
        # 1. add the 2 directions of encoder_hidden
        # encoder_out = (encoder_out[:, :, :self.hidden_dim] + encoder_out[:, :, self.hidden_dim:])

        # 2. return the last state of encoder_hidden
        # decoder_hidden = encoder_hidden[-1:]

        # best practice
        # 3. (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        decoder_hidden = torch.cat([encoder_hidden[0:encoder_hidden.size(0):2], encoder_hidden[1:encoder_hidden.size(0):2]], 2)
        return decoder_hidden

    def transform(self, input, mask):
        mask_token = torch.full(input[0].size(), self.rel_size, dtype=torch.long, device=self.args.device)
        SOT_token = torch.full(input[0].size(), self.rel_size+1, dtype=torch.long, device=self.args.device)
        EOT_token = torch.full(input[0].size(), self.rel_size+2, dtype=torch.long, device=self.args.device)

        if mask:
            enc_input = (mask_token, input[1], input[2], EOT_token)
        else:
            enc_input = (input[0], input[1], mask_token, EOT_token)
        dec_input = (SOT_token, input[0], input[1], input[2])
        target = torch.stack((input[0], input[1], input[2], EOT_token), dim=0).detach_()

        return enc_input, dec_input, target

    def forward(self, input, mask):
        enc_input, dec_input, target = self.transform(input, mask)

        output, hidden = self.encoder(enc_input)
        hidden = self.cat_hidden(hidden)
        output, hidden, logits, probs, entropy = self.decoder(dec_input, hidden, mask)

        loss = None
        if self.pretraining:
            loss = self.loss_function(logits, target)

        return target, output, loss, probs, entropy
