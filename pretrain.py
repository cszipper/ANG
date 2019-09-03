import os
import logging
import time
from termcolor import colored
import argparse
import datetime
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import random

from utils import KBDataset, DataChecker, Config

logging.basicConfig(
    level=logging.DEBUG,
    format='%(module)5s %(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)

parser = argparse.ArgumentParser(description='GAN for Knowledge Embedding')
parser.add_argument('--config', type=str, default="pretrain_wn18_50", metavar='S',
                    help='config file')
param = parser.parse_args()

args = Config(param.config)

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    torch.cuda.set_device(0)

logging.getLogger().addHandler(logging.FileHandler("log/" + args.perfix + datetime.datetime.now().strftime("%m%d%H%M%S")))

# init dataset
train_data = KBDataset(os.path.join("data", args.task_dir, 'train2id.txt'), args.cuda)
valid_data = KBDataset(os.path.join("data", args.task_dir, 'valid2id.txt'), args.cuda)
test_data = KBDataset(os.path.join("data", args.task_dir, 'test2id.txt'), args.cuda)
chk = DataChecker(args.task_dir)

# init data loader
train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_sampler = data.BatchSampler(data.RandomSampler(test_data), batch_size=args.sample_size, drop_last=True)

ent_size = chk.ent_size
rel_size = chk.rel_size

from models.seq2seq_egreedy import Seq2Seq

gen = Seq2Seq(args, rel_size, ent_size)

g_opt = None
d_opt = None
if args.optim == "SGD":
    g_opt = optim.SGD(gen.parameters(), lr=args.lr, momentum=0.9)
elif args.optim == "Adam":
    g_opt = optim.Adam(gen.parameters())

if args.cuda:
    gen.cuda()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
#
# fig.show()
# fig.canvas.draw()

# def make_mask(batch_size, currupt_rate):
#     mask = np.random.choice([True, False], size=[batch_size, 1], p=[currupt_rate, 1-currupt_rate])
#     return mask

def make_mask(currupt_rate):
    mask = True if random.random() < currupt_rate else False
    return mask

def pretty_print(pos, neg, mask, sample_batch_size):
    for i in range(sample_batch_size):
        psrc = '%6d' % pos[0][i] if not mask else colored(("%6d" % pos[0][i]), 'green')
        prel = '%6d' % pos[1][i]
        pdst = '%6d' % pos[2][i] if mask else colored(("%6d" % pos[2][i]), 'green')
        nsrc = '%6d' % neg[0][i] if not mask else colored(("%6d" % neg[0][i]), 'red')
        nrel = "%6d" % neg[1][i]
        ndst = '%6d' % neg[2][i] if mask else colored(("%6d" % neg[2][i]), 'red')
        logging.info('-- Case#%d', i)
        logging.info('-- Real: %s\t%s\t%s\t%6d', psrc, prel, pdst, pos[3][i])
        logging.info('-- Fake: %s\t%s\t%s\t%6d', nsrc, nrel, ndst, neg[3][i])

def pretrain():
    batches = len(train_data)/args.batch_size
    gen.train()
    criterion = nn.NLLLoss()
    loss_matrix = []
    valid_loss = []
    for epoch in range(args.epoches):
        epoch_loss = 0
        for batch_idx, pos in enumerate(train_loader):
            neg_mask = make_mask(args.currupt_rate)
            _, _, g_loss, _ = gen(pos, neg_mask, 1)

            gen.zero_grad()
            g_loss.backward()
            g_opt.step()

            epoch_loss += g_loss

        # logging
        avg_loss = epoch_loss / len(train_data)
        loss_matrix.append(avg_loss.detach_())
        # validing
        v_loss = valid().detach_()
        valid_loss.append(v_loss)
        # ax.clear()
        # ax.plot(loss_matrix, color="red")
        # ax.plot(test_loss, color="green")
        # fig.canvas.draw()
        logging.info('Epoch %d/%d, G_loss=%f, T_loss=%f', epoch + 1, args.epoches, avg_loss, v_loss)

    torch.save(gen.state_dict(), os.path.join("res", args.task_dir, args.perfix))
    torch.save(g_opt.state_dict(), os.path.join("res", args.task_dir, args.perfix + "_opt"))

def test():
    with torch.no_grad():
        gen.eval()
        epoch_loss = 0
        for batch_idx, pos in enumerate(test_loader):
            sample_batch_size = pos[0].size()[0]
            neg_mask = make_mask(args.currupt_rate)
            _, _, g_loss, _ = gen(pos, neg_mask, 1)
            epoch_loss += g_loss

        # logging
        avg_loss = epoch_loss / len(test_data)
    gen.train()
    return avg_loss

def valid():
    with torch.no_grad():
        gen.eval()
        epoch_loss = 0
        for batch_idx, pos in enumerate(valid_loader):
            sample_batch_size = pos[0].size()[0]
            neg_mask = make_mask(args.currupt_rate)
            _, _, g_loss, _ = gen(pos, neg_mask, 1)
            epoch_loss += g_loss

        # logging
        avg_loss = epoch_loss / len(valid_loader)
    gen.train()
    return avg_loss

def sample():
    with torch.no_grad():
        gen.eval()
        # get 10 sample
        idx, sample = next(enumerate(test_sampler))
        src = []
        rel = []
        dst = []
        for i in sample:
            src.append(test_data[i][0])
            rel.append(test_data[i][1])
            dst.append(test_data[i][2])
        src = torch.tensor(src).view(-1, 1)
        rel = torch.tensor(rel).view(-1, 1)
        dst = torch.tensor(dst).view(-1, 1)
        if args.cuda:
            src = src.cuda()
            rel = rel.cuda()
            dst = dst.cuda()
        sample = (src, rel, dst)
        neg_mask = make_mask(args.currupt_rate)
        target, output, _, _ = gen(sample, neg_mask, 1)

        logging.info("Sample generated triplets:")
        logging.info("-- Label\tsrc\trel\tdst\tEOT")
        pretty_print(target, output, neg_mask, args.sample_size)
        logging.info('===================================================================')
    gen.train()

if __name__ == '__main__':
    pretrain()
