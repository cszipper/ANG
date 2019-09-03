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
        self.args = args
        self.rel_emb = nn.Embedding(rel_size, args.latent_dim)
        self.ent_emb = nn.Embedding(ent_size, args.latent_dim)
        self.criterion = nn.MarginRankingLoss(self.margin, reduction="sum")
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)

    def constraint(self):
        self.rel_emb.weight.data.renorm_(2, 0, 1)
        self.ent_emb.weight.data.renorm_(2, 0, 1)

    def _calc(self,h,t,r):
        return torch.norm(h + r - t, 1, -1)

    def loss_func(self, p_score, n_score):
        y = torch.Tensor([-1])
        if self.args.cuda:
            y = y.cuda()
        loss = self.criterion(p_score, n_score, y)
        return loss

    def forward(self, pos, neg, take):
        [pos_h, pos_r, pos_t] = pos
        [neg_h, neg_r, neg_t] = neg
        p_h = self.ent_emb(pos_h)
        p_t = self.ent_emb(pos_t)
        p_r = self.rel_emb(pos_r)
        n_h = self.ent_emb(neg_h)
        n_t = self.ent_emb(neg_t)
        n_r = self.rel_emb(neg_r)
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        p_score = torch.sum(_p_score.masked_select(take).view(-1, 1), -1)
        n_score = torch.sum(_n_score.masked_select(take).view(-1, 1), -1)
        loss = self.loss_func(p_score, n_score)
        return loss, _n_score

    def predict(self, predict_h, predict_t, predict_r):
        v_h = torch.from_numpy(predict_h)
        v_t = torch.from_numpy(predict_t)
        v_r = torch.from_numpy(predict_r)
        if self.args.cuda:
            v_h = v_h.cuda()
            v_t = v_t.cuda()
            v_r = v_r.cuda()
        p_h = self.ent_emb(v_h)
        p_t = self.ent_emb(v_t)
        p_r = self.rel_emb(v_r)
        p_score = self._calc(p_h, p_t, p_r)
        return p_score.cpu()

    def load_param(self, filename):
        f = open(filename, "r")
        content = json.loads(f.read())
        f.close()
        self.state_dict().get("ent_emb.weight").copy_(torch.from_numpy(np.array(content["ent_embeddings.weight"])))
        self.state_dict().get("rel_emb.weight").copy_(torch.from_numpy(np.array(content["rel_embeddings.weight"])))
