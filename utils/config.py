import yaml

class Config(object):
    """docstring for Config."""
    def __init__(self, filename):
        f = yaml.load(open("config/"+filename, 'r'))
        self.pretrain = f['pretrain']
        self.batch_size = f['batch_size']
        self.epoches = f['epoches']
        self.latent_dim = f['latent_dim']
        self.lr = f['lr']
        self.epoch_per_test = f['epoch_per_test']
        self.optim = f['optim']
        self.dropout = f['dropout']
        self.keep_prob = f['keep_prob']
        self.currupt_rate = f['currupt_rate']
        self.sample_size = f['sample_size']
        self.no_cuda = f['no_cuda']
        self.seed = f['seed']
        self.task_dir = f['task_dir']
        self.perfix = f['perfix']
