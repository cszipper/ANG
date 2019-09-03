import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.rel_transfer=nn.Embedding(rel_size, args.latent_dim)
        self.ent_transfer=nn.Embedding(ent_size, args.latent_dim)
        self.criterion = nn.MarginRankingLoss(self.margin, reduction="sum")
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.rel_emb.weight.data)
        nn.init.xavier_uniform_(self.ent_emb.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer.weight.data)

    def constraint(self):
        self.rel_emb.weight.data.renorm_(2, 0, 1)
        self.ent_emb.weight.data.renorm_(2, 0, 1)

    def _transfer(self, e, t, r):
        return F.normalize(e + torch.sum(e * t, dim=-1, keepdim=True) * r, p=2, dim=-1)

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
        p_h_e = self.ent_emb(pos_h)
        p_t_e = self.ent_emb(pos_t)
        p_r_e = self.rel_emb(pos_r)
        n_h_e = self.ent_emb(neg_h)
        n_t_e = self.ent_emb(neg_t)
        n_r_e = self.rel_emb(neg_r)
        p_h_t = self.ent_transfer(pos_h)
        p_t_t = self.ent_transfer(pos_t)
        p_r_t = self.rel_transfer(pos_r)
        n_h_t = self.ent_transfer(neg_h)
        n_t_t = self.ent_transfer(neg_t)
        n_r_t = self.rel_transfer(neg_r)
        p_h = self._transfer(p_h_e,p_h_t,p_r_t)
        p_t = self._transfer(p_t_e,p_t_t,p_r_t)
        p_r = p_r_e
        n_h = self._transfer(n_h_e,n_h_t,n_r_t)
        n_t = self._transfer(n_t_e,n_t_t,n_r_t)
        n_r = n_r_e
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        p_score = torch.sum(_p_score.masked_select(take).view(-1, 1), -1)
        n_score = torch.sum(_n_score.masked_select(take).view(-1, 1), -1)
        loss = self.loss_func(p_score, n_score)
        return loss, -_n_score

    def predict(self, predict_h, predict_t, predict_r):
        v_h = torch.from_numpy(predict_h)
        v_t = torch.from_numpy(predict_t)
        v_r = torch.from_numpy(predict_r)
        if self.args.cuda:
            v_h = v_h.cuda()
            v_t = v_t.cuda()
            v_r = v_r.cuda()
        p_h_e = self.ent_emb(v_h)
        p_t_e = self.ent_emb(v_t)
        p_r_e = self.rel_emb(v_r)
        p_h_t = self.ent_transfer(v_h)
        p_t_t = self.ent_transfer(v_t)
        p_r_t = self.rel_transfer(v_r)
        p_h = self._transfer(p_h_e,p_h_t,p_r_t)
        p_t = self._transfer(p_t_e,p_t_t,p_r_t)
        p_r = p_r_e
        p_score = self._calc(p_h, p_t, p_r)
        return p_score.cpu()

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(module)5s %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )
    class Arg:
        latent_dim = 100
        cuda = False
        margin = 1.0
        device = "cpu"
    arg = Arg()
    dis = Discriminator(arg, 237, 14541)
    # dis.load_state_dict(
    #     torch.load(os.path.join("res", "fb15k237", "TransE-dis-pretrain"),
    #             map_location=lambda storage, loc: storage))
    dis.load_param("res/embedding.vec.json")
    dis.init_test(os.path.join("data", "fb15k237"))
    dis.test_link_prediction()
    dis.test_triple_classification()
