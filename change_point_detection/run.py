import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../models/Baselines')
sys.path.append('../common')
import torch
import torch.nn as nn
import numpy as np
from   itertools import chain 
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
from GLR_Hawkes import GLR_Hawkes
from GLR_Poisson import GLR_Poisson
from Score_statistics import Score_statistics
from BOCPD import BOCPD
from RBOCPD import RBOCPD
from CUSUM import CUSUM
from GLR_Hawkes_Multi import GLR_Hawkes_Multi
eps=sys.float_info.epsilon

class Opt:
    def __init__(self):
        self.underlying = "hawkes"#
        self.method = 'Score_statistics'
        self.path_dir = "../results/"+self.method+"/particle-data_temp"
        self.save_dir = self.path_dir+"/"#
        self.load_dir = "../data/particle-data_processed"
        self.num_changepoints = 1#
        self.multiple = True
        self.num_sequences = 1#
        self.safe = 1
        self.run_time = 100#
        self.window_size = None
        self.epochs = 400
        self.lr = 1e-1
        self.save_poisson_intensity=self.save_dir+'Discrete_intensity.pdf'
        self.poisson_interval=1.0 # TBD
        # self.d=4
        

if __name__ == '__main__':

    opt = Opt()
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    opt.device = torch.device('cpu')
    print("Device used for training: ",opt.device)
    # exit()

    # opt.save_dir = "../Results/Hawkes_synthetic_multiple_cp/BOCPD_null_dist_data_run_20000_seq_500"
    # opt.load_dir = "../data/Hawkes_synthetic_multiple_cp/null_dist_data_run_20000_seq_500"
        
    dataset, opt.d = load_dataset(opt.load_dir)   
    # exit() 
    if opt.method=='BOCPD':  
        cp_detector = BOCPD(opt=opt)
    if opt.method=='RBOCPD':  
        cp_detector = RBOCPD(opt=opt)
    if opt.method=='GLR_Hawkes':  
        cp_detector = GLR_Hawkes(opt=opt)
    if opt.method=='GLR_Poisson':  
        cp_detector = GLR_Poisson(opt=opt)
    if opt.method=='Score_statistics':  
        cp_detector = Score_statistics(opt=opt)
    if opt.method=='M_statistics':  
        cp_detector = Score_statistics(opt=opt) # TBD 
    if opt.method=='CUSUM':  
        cp_detector = CUSUM(opt=opt) 
    if opt.method=='GLR_Hawkes_Multi':  
        cp_detector = GLR_Hawkes_Multi(opt=opt,d=opt.d)

    results, Scores = [], []
    Scores_file = opt.save_dir + 'Scores'
    results_file = opt.save_dir + 'results'
    llratio_plot = opt.save_dir + 'llratio.jpg'
    if os.path.exists(opt.path_dir) == False :
        os.makedirs(opt.path_dir)

    for i,sequence in enumerate(dataset):
        # result, scores=cp_detector.DetectChangePoint(sequence,opt.save_dir)
        result, scores=cp_detector.DetectChangePoint(sequence,llratio_plot)
        # exit()
        results += [result]
        Scores += [scores]
        if i%10==0 or i == len(dataset)-1:
            save(results_file,results)
            save(Scores_file, Scores)