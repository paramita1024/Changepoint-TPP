import random
import sys
sys.path.append('../../common')
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
import torch
eps=sys.float_info.epsilon

class GLR_Poisson:

    def __init__(self, opt=None):
        self.L=15
        self.gamma=2
        self.MAX_ITER_MLE=50
        self.interval=2.0 # 
        self.opt=opt

    def DetectChangePoint(self, sequence):    
        LLRatio=self.ChangePointDetectionSequence(sequence) 
        # exit()
        # save('LLRatio', LLRatio)
        # exit()
        # LLRatio=load_data('LLRatio')
        # exit()
        # LLRatio_plot(sequence, LLRatio, 'LLRatio_Poisson.pdf')
        # exit()
        event_time,_,_,_, change_points_true = sequence
        finish_time = event_time[-1]
        cp_vector_true=GetCPVectorized(change_points_true, LLRatio)
        # exit()
        roc = ROC(LLRatio, cp_vector_true)
        # exit()
        change_points = ChangePointsFromLLRatio(LLRatio)
        # exit()
        detection_delay=GetDetectionDelay(change_points_true, \
            change_points, finish_time)       
        results={'roc':roc, 'change_points_estimated':change_points, \
        'change_points_true':cp_vector_true,\
        'detection_delay':detection_delay}
        # exit()
        return results, LLRatio

    def ChangePointDetectionSequence(self,sequence):
        event_time, event_type,intensity,intensity_times,_=sequence
        event_count = DiscretizePoisson(event_time,self.interval) 
        # Plot_time_count(intensity_times, intensity, event_count, self.opt.save_poisson_intensity)
        start_index, end_index, LLRatio = 0, self.L, []
        sub_event_count = event_count[start_index:end_index]
        # exit()
        mu_init = self.MLE_Poisson(sub_event_count)
        # exit()
        set_of_mu = [(start_index,end_index,mu_init)]
        start_index += self.L
        end_index = start_index+self.L
        # exit()
        while True:
            sub_event_count = event_count[start_index:end_index]
            mu_last_window =set_of_mu[-1][2]
            mu = self.MLE_Poisson(sub_event_count,mu_init_val=mu_last_window)
            # exit()
            set_of_mu.append((start_index,end_index,mu)) 
            # exit()
            mu_init=GetAlphaLastPartition(set_of_mu,start_index)
            # exit()
            LLRatio_window= self.GetLogLikelihoodRatio(mu, mu_init, \
             sub_event_count)
            # exit()
            LLRatio.append((start_index, end_index, LLRatio_window,\
                start_index*self.interval))
            start_index += self.gamma
            end_index = start_index+self.L
            # break
            if end_index > len(event_count):
                break
        # for l in LLRatio:
        #     print(l)
        return LLRatio

    def GetLogLikelihoodRatio(self,mu, mu_init, event_count):
        n_event_count=len(event_count)
        total_events = sum(event_count)
        return total_events*math.log(mu/mu_init)-n_event_count*(mu-mu_init)

    def MLE_Poisson(self, event_count, mu_init_val=None):
        if mu_init_val==None:
            mu_init=[random.random()]
        else:
            mu_init=[mu_init_val]
        t=len(event_count)
        n=sum(event_count)
        queue_length=50
        MAX_ITER_MLE=self.MAX_ITER_MLE
        def proj(x):
            return max(x,0)
        def f(x):
            if x==0:
                x+=0.00001
            return x*t-n*math.log(x)
        def grad(x):
            if x==0:
                x+=0.00001
            return t-float(n)/x 
        
        # exit()
        # MAX_ITER_MLE=3
        spg_obj = spg()  
        # exit()
        result= spg_obj.solve( mu_init, f, grad, proj, \
            queue_length , eps, MAX_ITER_MLE)
        # exit()
        mu=result['bestX'][0]
        # print(mu)
        # exit()
        return mu