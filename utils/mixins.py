import ctypes
import json
import logging
import numpy as np
import torch
import torch.nn as nn

class LPResults(ctypes.Structure):
    """docstring for LPResults."""
    _fields_ = [("l_reci_rank", ctypes.c_float),
                ("l_rank", ctypes.c_float),
                ("l_tot", ctypes.c_float),
                ("l3_tot", ctypes.c_float),
                ("l1_tot", ctypes.c_float),
                ("r_reci_rank", ctypes.c_float),
                ("r_rank", ctypes.c_float),
                ("r_tot", ctypes.c_float),
                ("r3_tot", ctypes.c_float),
                ("r1_tot", ctypes.c_float),
                ("a_reci_rank", ctypes.c_float),
                ("a_rank", ctypes.c_float),
                ("a_tot", ctypes.c_float),
                ("a3_tot", ctypes.c_float),
                ("a1_tot", ctypes.c_float),
                ("l_filter_reci_rank", ctypes.c_float),
                ("l_filter_rank", ctypes.c_float),
                ("l_filter_tot", ctypes.c_float),
                ("l3_filter_tot", ctypes.c_float),
                ("l1_filter_tot", ctypes.c_float),
                ("r_filter_reci_rank", ctypes.c_float),
                ("r_filter_rank", ctypes.c_float),
                ("r_filter_tot", ctypes.c_float),
                ("r3_filter_tot", ctypes.c_float),
                ("r1_filter_tot", ctypes.c_float),
                ("a_filter_reci_rank", ctypes.c_float),
                ("a_filter_rank", ctypes.c_float),
                ("a_filter_tot", ctypes.c_float),
                ("a3_filter_tot", ctypes.c_float),
                ("a1_filter_tot", ctypes.c_float),
                ("l_reci_rank_constrain", ctypes.c_float),
                ("l_rank_constrain", ctypes.c_float),
                ("l_tot_constrain", ctypes.c_float),
                ("l3_tot_constrain", ctypes.c_float),
                ("l1_tot_constrain", ctypes.c_float),
                ("r_reci_rank_constrain", ctypes.c_float),
                ("r_rank_constrain", ctypes.c_float),
                ("r_tot_constrain", ctypes.c_float),
                ("r3_tot_constrain", ctypes.c_float),
                ("r1_tot_constrain", ctypes.c_float),
                ("a_reci_rank_constrain", ctypes.c_float),
                ("a_rank_constrain", ctypes.c_float),
                ("a_tot_constrain", ctypes.c_float),
                ("a3_tot_constrain", ctypes.c_float),
                ("a1_tot_constrain", ctypes.c_float),
                ("l_filter_reci_rank_constrain", ctypes.c_float),
                ("l_filter_rank_constrain", ctypes.c_float),
                ("l_filter_tot_constrain", ctypes.c_float),
                ("l3_filter_tot_constrain", ctypes.c_float),
                ("l1_filter_tot_constrain", ctypes.c_float),
                ("r_filter_reci_rank_constrain", ctypes.c_float),
                ("r_filter_rank_constrain", ctypes.c_float),
                ("r_filter_tot_constrain", ctypes.c_float),
                ("r3_filter_tot_constrain", ctypes.c_float),
                ("r1_filter_tot_constrain", ctypes.c_float),
                ("a_filter_reci_rank_constrain", ctypes.c_float),
                ("a_filter_rank_constrain", ctypes.c_float),
                ("a_filter_tot_constrain", ctypes.c_float),
                ("a3_filter_tot_constrain", ctypes.c_float),
                ("a1_filter_tot_constrain", ctypes.c_float)]

class TestMixin:
    """ TestMixin for model evaluation."""
    def init_test(self, in_path):
        self.in_path = in_path + "/"

        self.lib = ctypes.cdll.LoadLibrary("./lib/Base.so")
        self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.test_link_prediction.restype = LPResults
        self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.test_triple_classification.restype = ctypes.c_float

        self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode('utf-8'), len(self.in_path) * 2))
        self.lib.randReset()
        self.lib.setWorkThreads(8)
        self.lib.importTrainFiles()
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.test_total = self.lib.getTestTotal()
        self.valid_total = self.lib.getValidTotal()

        # link prediction
        self.test_h = np.zeros(self.ent_size, dtype = np.int64)
        self.test_t = np.zeros(self.ent_size, dtype = np.int64)
        self.test_r = np.zeros(self.ent_size, dtype = np.int64)
        self.test_h_addr = self.test_h.__array_interface__['data'][0]
        self.test_t_addr = self.test_t.__array_interface__['data'][0]
        self.test_r_addr = self.test_r.__array_interface__['data'][0]

        # triple classification
        self.test_pos_h = np.zeros(self.test_total, dtype = np.int64)
        self.test_pos_t = np.zeros(self.test_total, dtype = np.int64)
        self.test_pos_r = np.zeros(self.test_total, dtype = np.int64)
        self.test_neg_h = np.zeros(self.test_total, dtype = np.int64)
        self.test_neg_t = np.zeros(self.test_total, dtype = np.int64)
        self.test_neg_r = np.zeros(self.test_total, dtype = np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
        self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]
        self.valid_pos_h = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_pos_t = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_pos_r = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_neg_h = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_neg_t = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_neg_r = np.zeros(self.valid_total, dtype = np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]
        self.relThresh = np.zeros(self.rel_size, dtype = np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

    def test_link_prediction(self):
        # hits@10 mean_rank
        for epoch in range(self.test_total):
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.predict(self.test_h, self.test_t, self.test_r)
            self.lib.testHead(res.data.numpy().__array_interface__['data'][0])

            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.predict(self.test_h, self.test_t, self.test_r)
            self.lib.testTail(res.data.numpy().__array_interface__['data'][0])
        res = self.lib.test_link_prediction()

        logging.info("no type constraint results:")
        logging.info("metric:\tMRR\tMR\thit@10\thit@3\thit@1")
        logging.info("l(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.l_reci_rank, res.l_rank, res.l_tot, res.l3_tot, res.l1_tot)
        logging.info("r(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.r_reci_rank, res.r_rank, res.r_tot, res.r3_tot, res.r1_tot)
        logging.info("averaged(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.a_reci_rank, res.a_rank, res.a_tot, res.a3_tot, res.a1_tot)
        logging.info("")
        logging.info("l(filter):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.l_filter_reci_rank, res.l_filter_rank, res.l_filter_tot, res.l3_filter_tot, res.l1_filter_tot)
        logging.info("r(filter):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.r_filter_reci_rank, res.r_filter_rank, res.r_filter_tot, res.r3_filter_tot, res.r1_filter_tot)
        logging.info("averaged(filter):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.a_filter_reci_rank, res.a_filter_rank, res.a_filter_tot, res.a3_filter_tot, res.a1_filter_tot)

        logging.info("type constraint results:")
        logging.info("metric:\tMRR\tMR\thit@10\thit@3\thit@1")
        logging.info("l(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.l_reci_rank_constrain, res.l_rank_constrain, res.l_tot_constrain, res.l3_tot_constrain, res.l1_tot_constrain)
        logging.info("r(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.r_reci_rank_constrain, res.r_rank_constrain, res.r_tot_constrain, res.r3_tot_constrain, res.r1_tot_constrain)
        logging.info("averaged(raw):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.a_reci_rank_constrain, res.a_rank_constrain, res.a_tot_constrain, res.a3_tot_constrain, res.a1_tot_constrain)
        logging.info("")
        logging.info("l(filter):\t %6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.l_filter_reci_rank_constrain, res.l_filter_rank_constrain, res.l_filter_tot_constrain, res.l3_filter_tot_constrain, res.l1_filter_tot_constrain)
        logging.info("r(filter):\t %6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.r_filter_reci_rank_constrain, res.r_filter_rank_constrain, res.r_filter_tot_constrain, res.r3_filter_tot_constrain, res.r1_filter_tot_constrain)
        logging.info("averaged(filter):\t%6.6f\t%6.6f\t%6.6f\t%6.6f\t%6.6f",
                res.a_filter_reci_rank_constrain, res.a_filter_rank_constrain, res.a_filter_tot_constrain, res.a3_filter_tot_constrain, res.a1_filter_tot_constrain)

        self.lib.restore_link_prediction()

        return res

    def test_triple_classification(self):
        # hits@10 mean_rank
        self.lib.getValidBatch(
            self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr,
            self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr
        )
        res_pos = self.predict(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
        res_neg = self.predict(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
        self.lib.getBestThreshold(
            self.relThresh_addr, res_pos.data.numpy().__array_interface__['data'][0],
            res_neg.data.numpy().__array_interface__['data'][0]
        )

        self.lib.getTestBatch(
            self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr,
            self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr
        )
        res_pos = self.predict(self.test_pos_h, self.test_pos_t, self.test_pos_r)
        res_neg = self.predict(self.test_neg_h, self.test_neg_t, self.test_neg_r)
        res = self.lib.test_triple_classification(
            self.relThresh_addr, res_pos.data.numpy().__array_interface__['data'][0],
            res_neg.data.numpy().__array_interface__['data'][0]
        )
        logging.info("triple classification accuracy is %lf", res)
        return res

class PretrainMixin:
    """ Load pretrained parameters """
    def load_param(self, filename):
        f = open(filename, "r")
        content = json.loads(f.read())
        f.close()
        self.state_dict().get("ent_emb.weight").copy_(torch.from_numpy(np.array(content["model.ent_emb.weight"])))
        self.state_dict().get("rel_emb.weight").copy_(torch.from_numpy(np.array(content["model.rel_emb.weight"])))
