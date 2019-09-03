import logging
import time
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir="log/"+time.strftime("GAN-%d-%H-%M", time.localtime()))

def add_scalar(epoch, lp, tc):
    writer.add_scalars('lp/no_constraint/MRR', {
    #                                'l_reci_rank': lp.l_reci_rank,
    #                                'r_reci_rank': lp.r_reci_rank,
                                   'a_reci_rank': lp.a_reci_rank,
    #                                'l_filter_reci_rank': lp.l_filter_reci_rank,
    #                                'r_filter_reci_rank': lp.r_filter_reci_rank,
                                   'a_filter_reci_rank': lp.a_filter_reci_rank
                                 }, epoch)
    writer.add_scalars('lp/no_constraint/MR', {
                                   # 'l_rank': lp.l_rank,
                                   # 'r_rank': lp.r_rank,
                                   'a_rank': lp.a_rank,
                                   # 'l_filter_rank': lp.l_filter_rank,
                                   # 'r_filter_rank': lp.r_filter_rank,
                                   'a_filter_rank': lp.a_filter_rank
                                 }, epoch)
    writer.add_scalars('lp/no_constraint/hits_10', {
                                   # 'l_tot': lp.l_tot,
                                   # 'r_tot': lp.r_tot,
                                   'a_tot': lp.a_tot,
                                   # 'l_filter_tot': lp.l_filter_tot,
                                   # 'r_filter_tot': lp.r_filter_tot,
                                   'a_filter_tot': lp.a_filter_tot
                                 }, epoch)
    # writer.add_scalars('lp/no_constraint/hits_3', {
    #                                'l3_tot': lp.l3_tot,
    #                                'r3_tot': lp.r3_tot,
    #                                'a3_tot': lp.a3_tot,
    #                                'l3_filter_tot': lp.l3_filter_tot,
    #                                'r3_filter_tot': lp.r3_filter_tot,
    #                                'a3_filter_tot': lp.a3_filter_tot
    #                              }, epoch)
    # writer.add_scalars('lp/no_constraint/hits_1', {
    #                                'l1_tot': lp.l1_tot,
    #                                'r1_tot': lp.r1_tot,
    #                                'a1_tot': lp.a1_tot,
    #                                'l1_filter_tot': lp.l1_filter_tot,
    #                                'r1_filter_tot': lp.r1_filter_tot,
    #                                'a1_filter_tot': lp.a1_filter_tot
    #                              }, epoch)
    writer.add_scalars('lp/with_constraint/MRR', {
    #                                'l_reci_rank_constrain': lp.l_reci_rank_constrain,
    #                                'r_reci_rank_constrain': lp.r_reci_rank_constrain,
                                   'a_reci_rank_constrain': lp.a_reci_rank_constrain,
    #                                'l_filter_reci_rank_constrain': lp.l_filter_reci_rank_constrain,
    #                                'r_filter_reci_rank_constrain': lp.r_filter_reci_rank_constrain,
                                   'a_filter_reci_rank_constrain': lp.a_filter_reci_rank_constrain
                                 }, epoch)
    writer.add_scalars('lp/with_constraint/MR', {
                                   # 'l_rank_constrain': lp.l_rank_constrain,
                                   # 'r_rank_constrain': lp.r_rank_constrain,
                                   'a_rank_constrain': lp.a_rank_constrain,
                                   # 'l_filter_rank_constrain': lp.l_filter_rank_constrain,
                                   # 'r_filter_rank_constrain': lp.r_filter_rank_constrain,
                                   'a_filter_rank_constrain': lp.a_filter_rank_constrain
                                 }, epoch)
    writer.add_scalars('lp/with_constraint/hits_10', {
                                   # 'l_tot_constrain': lp.l_tot_constrain,
                                   # 'r_tot_constrain': lp.r_tot_constrain,
                                   'a_tot_constrain': lp.a_tot_constrain,
                                   # 'l_filter_tot_constrain': lp.l_filter_tot_constrain,
                                   # 'r_filter_tot_constrain': lp.r_filter_tot_constrain,
                                   'a_filter_tot_constrain': lp.a_filter_tot_constrain
                                 }, epoch)
    # writer.add_scalars('lp/with_constraint/hits_3', {
    #                                'l3_tot_constrain': lp.l3_tot_constrain,
    #                                'r3_tot_constrain': lp.r3_tot_constrain,
    #                                'a3_tot_constrain': lp.a3_tot_constrain,
    #                                'l3_filter_tot_constrain': lp.l3_filter_tot_constrain,
    #                                'r3_filter_tot_constrain': lp.r3_filter_tot_constrain,
    #                                'a3_filter_tot_constrain': lp.a3_filter_tot_constrain
    #                              }, epoch)
    # writer.add_scalars('lp/with_constraint/hits_1', {
    #                                'l1_tot_constrain': lp.l1_tot_constrain,
    #                                'r1_tot_constrain': lp.r1_tot_constrain,
    #                                'a1_tot_constrain': lp.a1_tot_constrain,
    #                                'l1_filter_tot_constrain': lp.l1_filter_tot_constrain,
    #                                'r1_filter_tot_constrain': lp.r1_filter_tot_constrain,
    #                                'a1_filter_tot_constrain': lp.a1_filter_tot_constrain
    #                              }, epoch)
    writer.add_scalar('tc/accuracy', tc, epoch)
