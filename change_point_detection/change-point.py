import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../models/cvxpy')
sys.path.append('../common')


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import pandas as pd

from   preprocess.Dataset import MakeData
from   transformer.Models import Transformer
import common.Utils as Utils
from model import Model

class Opt:
    def __init__(self):
        self.underlying = "hawkes"
        self.model = "hawkes"
        self.save_dir = "../results/cvxpy_temp/exp_particle-data"
        self.load_file = "../data/particle-data_processed.pkl"
        self.safe = 1
        self.epochs = 100
        self.lr = 3e-2
        #self.save_file = "particle-data_processed.pkl"
        self.log_file = self.save_dir + ".txt"
        

        self.d_model = 16
        self.d_hid = 16
        self.n_layers = 1
        self.n_head = 1
        self.dropout = 0.1
        self.d_k = 16
        self.d_v = 16
        self.future = 10
        self.smooth = -0.1

        


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--load-file', required=True)
    # parser.add_argument('--save-dir', required=True)
    # parser.add_argument('--num-changepoints', type=int, required=True)
    # parser.add_argument('--lr', type=float, default=3e-2)
    # parser.add_argument('--epochs', type=int, default=40)
    # parser = parser.parse_args()
    
    opt = Opt()
    # opt.load_file = parser.load_file
    # opt.save_dir = parser.save_dir
    # opt.num_changepoints = parser.num_changepoints
    # opt.epochs = parser.epochs

    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    # opt.device = torch.device('cpu')
    print("Device used for training: ",opt.device)
    # logging
    with open(opt.log_file, 'w') as f:
        f.write('Device used for training: {}\n'.format(opt.device))

    # real_data = get_particle_data(opt.load_file)
    # data = MakeData(
    #     num_changepoints=opt.num_changepoints,
    #     num_sequences=opt.num_sequences
    # )
    data_dict = Utils.load_data(opt.load_file)
    # real_data = get_adele_data(data_dict)
    data_dict = Utils.preprocess_real_data(data_dict)
    # data_dict = data.render()
    # data.show(fig=True)
    # Utils.save(opt.save_file,data_dict)

    len_seq = len(data_dict["time"][:,0])
    len_feat = len(data_dict["features"][0])
    num_types = data_dict['num_types']
    model = Model(len_seq, num_types, len_feat, opt).to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('[Info] Number of parameters in model: {}'.format(num_params))
    # # logging
    # with open(opt.log_file, 'a') as f:
    #     f.write('[Info] Number of parameters in model: {}\n'.format(num_params))
    changes = model.train_me(
        data_dict=data_dict, 
        num_epochs=opt.epochs, 
        num_changes=opt.num_changepoints,
        model_type = opt.dist_type,
        optimizer=optimizer,
        device = opt.device,
        save_dir=opt.save_dir
    )

    # fig, axs = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)
    # n_bins = 50
    # axs.hist(np.array(changes), range=[0,opt.run_time], bins = n_bins)
    # if opt.save_dir is not None:
    #     plt.savefig(os.path.join(opt.save_dir, "change_points_histogram"))
    
