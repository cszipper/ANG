import os
import logging
import time
import random
import math

import torch
import torch.nn as nn
import numpy as np

from utils.logger import writer, add_scalar

EPS_START = 1.0
EPS_END = 0.4
EPS_DECAY = 250

class Trainer(object):
    """docstring for Trainer."""
    def __init__(self, args, gen, dis, g_opt, d_opt, chk, train_loader, train_data):
        super(Trainer, self).__init__()
        self.args = args
        self.gen = gen
        self.dis = dis
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.chk = chk
        self.train_loader = train_loader
        self.train_data = train_data

    def make_mask(self, currupt_rate):
        mask = True if random.random() < currupt_rate else False
        return mask

    def test(self, epoch):
        with torch.no_grad():
            self.gen.eval()
            self.dis.eval()
            lp = self.dis.test_link_prediction()
            tc = self.dis.test_triple_classification()
            add_scalar(epoch, lp, tc)

    def penalty(self, rewards, flags):
        r = rewards.cpu().detach_().numpy()
        r[flags.astype(np.int)] = self.args.penalty
        res = torch.Tensor(r)
        if self.args.cuda:
            return res.cuda()
        return res

    def contrib(self, flags, size):
        one = np.ones(size)
        one[flags] = 0
        take = torch.from_numpy(one).byte().view(-1, 1)
        if self.args.cuda:
            take = take.cuda()
        return take

    def get_threshold(self, epoch):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (epoch + 1) / EPS_DECAY)
        return eps_threshold
#         return 0.8

    # def train(self):
    #     avg_reward = 0
    #     total_run = 0
    #     args = param
    #     for epoch in range(args.epoches):
    #         gen.train()
    #         dis.train()
    #         epoch_loss = 0
    #         epoch_reward = 0
    #         for batch_idx, pos in enumerate(train_loader):
    #             neg_mask = make_mask(args.currupt_rate)
    #             _, neg, _, probs = gen(pos, neg_mask)
    #
    #             flags = chk.check(neg)
    #             percentage = len(flags)/args.batch_size
    #             writer.add_scalar("all/percentage", percentage, total_run)
    #             take = contrib(flags, pos[0].size(0))
    #
    #             d_opt.zero_grad()
    #             d_loss, rewards = dis(pos, neg, take)
    #             d_loss.backward()
    #             d_opt.step()
    #             dis.constraint()
    #
    #             rewards = penalty(rewards, flags)
    #             total_run += 1
    #
    #             g_opt.zero_grad()
    #             if batch_idx % args.n_critic == 0:
    #                 g_loss = gen.policy_gradient(rewards - avg_reward, probs.squeeze())
    #                 g_loss.backward()
    #                 nn.utils.clip_grad_norm_(gen.parameters(), 5)
    #                 g_opt.step()
    #
    #             epoch_loss += d_loss.item()
    #             epoch_reward += torch.sum(rewards).item()
    #
    #         # logging
    #         avg_loss = epoch_loss / len(train_data)
    #         writer.add_scalar('all/avg_loss', avg_loss, epoch)
    #         avg_reward = epoch_reward / len(train_data)
    #         writer.add_scalar('all/avg_reward', avg_reward, epoch)
    #         logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, args.epoches, avg_loss, avg_reward)
    #         # testing
    #         if (epoch + 1) % args.epoch_per_test == 0:
    #             test(epoch)

    def train_entropy(self):
        avg_reward = 0
        total_run = 0
        for epoch in range(self.args.epoches):
            self.gen.train()
            self.dis.train()
            epoch_loss = 0
            epoch_reward = 0

            for batch_idx, pos in enumerate(self.train_loader):
                neg_mask = self.make_mask(self.args.currupt_rate)
                _, neg, _, probs, entropy = self.gen(pos, neg_mask)

                flags = self.chk.check(neg)
                percentage = len(flags)/self.args.batch_size
                writer.add_scalar("all/percentage", percentage, total_run+1)
                writer.add_scalar("all/entropy", torch.mean(entropy.clone().cpu()), total_run+1)
                take = self.contrib(flags, pos[0].size(0))

                self.d_opt.zero_grad()
                d_loss, rewards = self.dis(pos, neg, take)
                d_loss.backward()
                self.d_opt.step()
                self.dis.constraint()

                rewards = self.penalty(rewards, flags)
                total_run += 1

                self.g_opt.zero_grad()
                if (batch_idx + 1) % self.args.n_critic == 0:
                    g_loss = self.gen.policy_gradient(rewards - avg_reward, probs.squeeze(), entropy)#, alpha=alpha)
                    g_loss.backward()
                    nn.utils.clip_grad_norm_(self.gen.parameters(), 5)
                    self.g_opt.step()

                epoch_loss += d_loss.item()
                epoch_reward += torch.sum(rewards).item()

            # logging
            avg_loss = epoch_loss / len(self.train_data)
            writer.add_scalar('all/avg_loss', avg_loss, epoch+1)
            avg_reward = epoch_reward / len(self.train_data)
            writer.add_scalar('all/avg_reward', avg_reward, epoch+1)
            logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, self.args.epoches, avg_loss, avg_reward)
            # testing
            if (epoch + 1) % self.args.epoch_per_test == 0:
                self.test(epoch)

    def train_greedy(self):
        avg_reward = 0
        total_run = 0
        for epoch in range(self.args.epoches):
            self.gen.train()
            self.dis.train()
            epoch_loss = 0
            epoch_reward = 0
            eps_threshold = self.get_threshold(epoch)

            for batch_idx, pos in enumerate(self.train_loader):
                neg_mask = self.make_mask(self.args.currupt_rate)
                _, neg, _, probs = self.gen(pos, neg_mask, eps_threshold)

                flags = self.chk.check(neg)
                percentage = len(flags)/self.args.batch_size
                writer.add_scalar("all/percentage", percentage, total_run+1)
                take = self.contrib(flags, pos[0].size(0))

                self.d_opt.zero_grad()
                d_loss, rewards = self.dis(pos, neg, take)
                d_loss.backward()
                self.d_opt.step()
                self.dis.constraint()

                rewards = self.penalty(rewards, flags)
                total_run += 1

                self.g_opt.zero_grad()
                if (batch_idx + 1) % self.args.n_critic == 0:
                    g_loss = self.gen.policy_gradient(rewards - avg_reward, probs.squeeze())
                    g_loss.backward()
                    nn.utils.clip_grad_norm_(self.gen.parameters(), 5)
                    self.g_opt.step()

                epoch_loss += d_loss.item()
                epoch_reward += torch.sum(rewards).item()

            # logging
            avg_loss = epoch_loss / len(self.train_data)
            writer.add_scalar('all/avg_loss', avg_loss, epoch+1)
            avg_reward = epoch_reward / len(self.train_data)
            writer.add_scalar('all/avg_reward', avg_reward, epoch+1)
            logging.info('Epoch %d/%d, D_loss=%f, reward=%f', epoch + 1, self.args.epoches, avg_loss, avg_reward)
            # testing
            if (epoch + 1) % self.args.epoch_per_test == 0:
                self.test(epoch)
