import os, sys
from collections import namedtuple, defaultdict
import pandas

def make_id(kb_index, task_dir):
    # generate rel2id ent2id train2id test2id valid2id
    with open(os.path.join("../data", task_dir, "relation2id.txt"), "w") as fp:
        fp.write(str(len(kb_index.rel_list)) + "\n")
        for name, idx in kb_index.rel_id.items():
            fp.write(name + " " + str(idx) + "\n")

    with open(os.path.join("../data", task_dir, "entity2id.txt"), "w") as fp:
        fp.write(str(len(kb_index.ent_list)) + "\n")
        for name, idx in kb_index.ent_id.items():
            fp.write(name + " " + str(idx) + "\n")

    with open(os.path.join("../data", task_dir, "train2id.txt"), "w") as fpw:
        with open(os.path.join("../data", task_dir, "train.txt"), "r") as fpr:
            lines = fpr.readlines()
            fpw.write(str(len(lines))+"\n")
            for line in lines:
                src, rel, dst = line.strip().split()
                fpw.write(
                    str(kb_index.ent_id[src]) + " " +
                    str(kb_index.ent_id[dst]) + " " +
                    str(kb_index.rel_id[rel]) + "\n"
                )

    with open(os.path.join("../data", task_dir, "test2id.txt"), "w") as fpw:
        with open(os.path.join("../data", task_dir, "test.txt"), "r") as fpr:
            lines = fpr.readlines()
            fpw.write(str(len(lines))+"\n")
            for line in lines:
                src, rel, dst = line.strip().split()
                fpw.write(
                    str(kb_index.ent_id[src]) + " " +
                    str(kb_index.ent_id[dst]) + " " +
                    str(kb_index.rel_id[rel]) + "\n"
                )

    with open(os.path.join("../data", task_dir, "valid2id.txt"), "w") as fpw:
        with open(os.path.join("../data", task_dir, "valid.txt"), "r") as fpr:
            lines = fpr.readlines()
            fpw.write(str(len(lines))+"\n")
            for line in lines:
                src, rel, dst = line.strip().split()
                fpw.write(
                    str(kb_index.ent_id[src]) + " " +
                    str(kb_index.ent_id[dst]) + " " +
                    str(kb_index.rel_id[rel]) + "\n"
                )

def make_constrain(task_dir):
    lef = {}
    rig = {}
    rellef = {}
    relrig = {}

    triple = open(os.path.join("../data", task_dir, "train2id.txt"), "r")
    valid = open(os.path.join("../data", task_dir, "valid2id.txt"), "r")
    test = open(os.path.join("../data", task_dir, "test2id.txt"), "r")

    tot = (int)(triple.readline())
    for i in range(tot):
    	content = triple.readline()
    	h,t,r = content.strip().split()
    	if not (h,r) in lef:
    		lef[(h,r)] = []
    	if not (r,t) in rig:
    		rig[(r,t)] = []
    	lef[(h,r)].append(t)
    	rig[(r,t)].append(h)
    	if not r in rellef:
    		rellef[r] = {}
    	if not r in relrig:
    		relrig[r] = {}
    	rellef[r][h] = 1
    	relrig[r][t] = 1

    tot = (int)(valid.readline())
    for i in range(tot):
    	content = valid.readline()
    	h,t,r = content.strip().split()
    	if not (h,r) in lef:
    		lef[(h,r)] = []
    	if not (r,t) in rig:
    		rig[(r,t)] = []
    	lef[(h,r)].append(t)
    	rig[(r,t)].append(h)
    	if not r in rellef:
    		rellef[r] = {}
    	if not r in relrig:
    		relrig[r] = {}
    	rellef[r][h] = 1
    	relrig[r][t] = 1

    tot = (int)(test.readline())
    for i in range(tot):
    	content = test.readline()
    	h,t,r = content.strip().split()
    	if not (h,r) in lef:
    		lef[(h,r)] = []
    	if not (r,t) in rig:
    		rig[(r,t)] = []
    	lef[(h,r)].append(t)
    	rig[(r,t)].append(h)
    	if not r in rellef:
    		rellef[r] = {}
    	if not r in relrig:
    		relrig[r] = {}
    	rellef[r][h] = 1
    	relrig[r][t] = 1

    test.close()
    valid.close()
    triple.close()

    f = open(os.path.join("../data", task_dir, "type_constrain.txt"), "w")
    f.write("%d\n"%(len(rellef)))
    for i in rellef:
    	f.write("%s\t%d"%(i,len(rellef[i])))
    	for j in rellef[i]:
    		f.write("\t%s"%(j))
    	f.write("\n")
    	f.write("%s\t%d"%(i,len(relrig[i])))
    	for j in relrig[i]:
    		f.write("\t%s"%(j))
    	f.write("\n")
    f.close()

    rellef = {}
    totlef = {}
    relrig = {}
    totrig = {}

    for i in lef:
    	if not i[1] in rellef:
    		rellef[i[1]] = 0
    		totlef[i[1]] = 0
    	rellef[i[1]] += len(lef[i])
    	totlef[i[1]] += 1.0

    for i in rig:
    	if not i[0] in relrig:
    		relrig[i[0]] = 0
    		totrig[i[0]] = 0
    	relrig[i[0]] += len(rig[i])
    	totrig[i[0]] += 1.0

    s11=0
    s1n=0
    sn1=0
    snn=0
    f = open(os.path.join("../data", task_dir, "test2id.txt"), "r")
    tot = (int)(f.readline())
    for i in range(tot):
    	content = f.readline()
    	h,t,r = content.strip().split()
    	rign = rellef[r] / totlef[r]
    	lefn = relrig[r] / totrig[r]
    	if (rign <= 1.5 and lefn <= 1.5):
    		s11+=1
    	if (rign > 1.5 and lefn <= 1.5):
    		s1n+=1
    	if (rign <= 1.5 and lefn > 1.5):
    		sn1+=1
    	if (rign > 1.5 and lefn > 1.5):
    		snn+=1
    f.close()


    f = open(os.path.join("../data", task_dir, "test2id.txt"), "r")
    f11 = open(os.path.join("../data", task_dir, "1-1.txt"), "w")
    f1n = open(os.path.join("../data", task_dir, "1-n.txt"), "w")
    fn1 = open(os.path.join("../data", task_dir, "n-1.txt"), "w")
    fnn = open(os.path.join("../data", task_dir, "n-n.txt"), "w")
    fall = open(os.path.join("../data", task_dir, "test2id_all.txt"), "w")
    tot = (int)(f.readline())
    fall.write("%d\n"%(tot))
    f11.write("%d\n"%(s11))
    f1n.write("%d\n"%(s1n))
    fn1.write("%d\n"%(sn1))
    fnn.write("%d\n"%(snn))
    for i in range(tot):
    	content = f.readline()
    	h,t,r = content.strip().split()
    	rign = rellef[r] / totlef[r]
    	lefn = relrig[r] / totrig[r]
    	if (rign <= 1.5 and lefn <= 1.5):
    		f11.write(content)
    		fall.write("0"+"\t"+content)
    	if (rign > 1.5 and lefn <= 1.5):
    		f1n.write(content)
    		fall.write("1"+"\t"+content)
    	if (rign <= 1.5 and lefn > 1.5):
    		fn1.write(content)
    		fall.write("2"+"\t"+content)
    	if (rign > 1.5 and lefn > 1.5):
    		fnn.write(content)
    		fall.write("3"+"\t"+content)
    fall.close()
    f.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()

class DataChecker(object):
    """docstring for DataChecker."""
    def __init__(self, data_dir):
        super(DataChecker, self).__init__()
        self.dataset = None
        self.ent_size = 0
        self.rel_size = 0
        self.load(data_dir)

    def load(self, data_dir):
        train = pandas.read_csv(
            os.path.join("data", data_dir, 'train2id.txt'), sep=" ", header=None, index_col=None, skiprows=[0])
        test = pandas.read_csv(
            os.path.join("data", data_dir, 'test2id.txt'), sep=" ", header=None, index_col=None, skiprows=[0])
        valid = pandas.read_csv(
            os.path.join("data", data_dir, 'valid2id.txt'), sep=" ", header=None, index_col=None, skiprows=[0])
        data = pandas.concat([train, test, valid], ignore_index=True)
        data.columns = ["src", "dst", "rel"]
        with open(os.path.join("data", data_dir, 'entity2id.txt')) as fp:
            self.ent_size = int(fp.readline().strip())
        with open(os.path.join("data", data_dir, 'relation2id.txt')) as fp:
            self.rel_size = int(fp.readline().strip())
        self.data = data.apply(tuple, axis="columns")

    def check(self, inputs):
        if inputs[0].is_cuda:
            triple = (inputs[0].squeeze().cpu(), inputs[1].squeeze().cpu(), inputs[2].squeeze().cpu())
        else:
            triple = (inputs[0].squeeze(), inputs[1].squeeze(), inputs[2].squeeze())
        src = triple[0].numpy()
        rel = triple[1].numpy()
        dst = triple[2].numpy()
        query = pandas.DataFrame({"src": src, "rel": rel, "dst":dst}, columns=["src", "dst", "rel"])
        res = query[query.apply(tuple, axis="columns").isin(self.data)]
        return res.index.values

if __name__ == '__main__':
    from data_loader import index_ent_rel, KBDataset
    from torch.utils.data import BatchSampler, RandomSampler, DataLoader
    task_dir = "test"
    # load data
    kb_index = index_ent_rel(os.path.join("../data", task_dir, 'train.txt'),
                             os.path.join("../data", task_dir, 'valid.txt'),
                             os.path.join("../data", task_dir, 'test.txt'))
    make_id(kb_index, task_dir)
#     test_data = KBDataset(os.path.join("../data", task_dir, 'test.txt'), kb_index, False)
#     test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

#     idx, sample = next(enumerate(test_loader))
#     chk = DataChecker(task_dir)
#     print(chk.check(sample))
