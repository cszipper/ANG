import numpy as np
import torch
import torch.nn as nn
from utils import TestMixin, PretrainMixin

class Discriminator(nn.Module, TestMixin, PretrainMixin):
    """ TransE based Discriminator. """
    def __init__(self, args, rel_size, ent_size):
        super(Discriminator, self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.margin = args.margin
        self.lmbda = 0.1
        self.args = args
        self.ent_re_embeddings=nn.Embedding(ent_size, args.latent_dim)
        self.ent_im_embeddings=nn.Embedding(ent_size, args.latent_dim)
        self.rel_re_embeddings=nn.Embedding(rel_size, args.latent_dim)
        self.rel_im_embeddings=nn.Embedding(rel_size, args.latent_dim)
        self.criterion = nn.Softplus().cuda()
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def constraint(self):
        pass

    def _calc(self, e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im):
        return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, dim=-1, keepdim=False)

    def loss_func(self, loss, regul):
        return torch.mean(loss + self.lmbda * regul)

    def forward(self, pos, neg, take):
        [pos_h, pos_r, pos_t] = pos
        [neg_h, neg_r, neg_t] = neg
        batch_size = pos_h.size(0)
        y = torch.cat((torch.ones(batch_size), -torch.ones(batch_size))).view(-1, 1)
        if self.args.cuda:
            y = y.cuda()
        batch_h = torch.cat((pos_h, neg_h))
        batch_r = torch.cat((pos_r, neg_r))
        batch_t = torch.cat((pos_t, neg_t))
        e_re_h=self.ent_re_embeddings(batch_h)
        e_im_h=self.ent_im_embeddings(batch_h)
        e_re_t=self.ent_re_embeddings(batch_t)
        e_im_t=self.ent_im_embeddings(batch_t)
        r_re=self.rel_re_embeddings(batch_r)
        r_im=self.rel_im_embeddings(batch_r)

        _score = self._calc(e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im)
        n_score = _score[batch_size:]
        score = (-y * _score).masked_select(torch.cat((take, take))).view(-1, 1)

        loss = self.criterion(score)
        regul = torch.mean(e_re_h**2) + torch.mean(e_im_h**2) + torch.mean(e_re_t**2) +\
                torch.mean(e_im_t**2) + torch.mean(r_re**2) + torch.mean(r_im**2)
        loss = self.loss_func(loss, regul)
        return loss, n_score

    def predict(self, predict_h, predict_t, predict_r):
        v_h = torch.from_numpy(predict_h)
        v_t = torch.from_numpy(predict_t)
        v_r = torch.from_numpy(predict_r)
        if self.args.cuda:
            v_h = v_h.cuda()
            v_t = v_t.cuda()
            v_r = v_r.cuda()
        p_re_h = self.ent_re_embeddings(v_h)
        p_re_t = self.ent_re_embeddings(v_t)
        p_re_r = self.rel_re_embeddings(v_r)
        p_im_h = self.ent_im_embeddings(v_h)
        p_im_t = self.ent_im_embeddings(v_t)
        p_im_r = self.rel_im_embeddings(v_r)
        p_score = -self._calc(p_re_h, p_im_h, p_re_t, p_im_t, p_re_r, p_im_r)
        return p_score.cpu()
