import os
import logging
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np

from utils import KBDataset, DataChecker, Config

logging.basicConfig(
    level=logging.DEBUG,
#    filename="log/log-pretrain.txt",
    format='%(module)5s %(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)

parser = argparse.ArgumentParser(description='GAN for Knowledge Embedding')
parser.add_argument('--pretrain', type=bool, default=False, metavar='B',
                    help='pretrain')
parser.add_argument('--config', type=str, default="gan_fb15k237_transE", metavar='S',
                    help='config file')
args = parser.parse_args()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.device = torch.device("cuda" if args.cuda else "cpu")

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

from models import Discriminator
from models.seq2seq_entropy import Seq2Seq as Generator
from trainer import Trainer

gen = Generator(args, rel_size, ent_size)
dis = Discriminator(args, rel_size, ent_size)
dis.init_test(os.path.join("data", args.task_dir))

g_opt = optim.SGD(gen.parameters(), lr=args.lr, momentum=0.9)
d_opt = optim.Adam(dis.parameters())

if args.cuda:
    gen.cuda()
    dis.cuda()

gen.load_state_dict(
    torch.load(os.path.join("res", args.task_dir, "TransE-gen-pretrain-dropout"),
            map_location=lambda storage, loc: storage))
# g_opt.load_state_dict(torch.load(os.path.join("res", args.task_dir, "TransE-gen-opt-pretrain-dropout"),
#             map_location=lambda storage, loc: storage))
# dis.load_state_dict(
#     torch.load(os.path.join("res", args.task_dir, "TransE-dis-pretrain"),
#             map_location=lambda storage, loc: storage))
trainer = Trainer(args, gen, dis, g_opt, d_opt, chk, train_loader)
trainer.train_entropy()

torch.save(
    gen.state_dict(),
    os.path.join("res", args.task_dir, "TransE-gen-"+time.strftime("%Y-%m-%d-%H%M%S", time.localtime())))
torch.save(
    dis.state_dict(),
    os.path.join("res", args.task_dir, "TransE-dis-"+time.strftime("%Y-%m-%d-%H%M%S", time.localtime())))
