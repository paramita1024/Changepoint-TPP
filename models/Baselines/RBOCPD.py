from __future__ import division
import sys
sys.path.append('../../common')
import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from Utils_baselines import *


"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection original version

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""
class RBOCPD:
    def __init__(self, opt=None):

        self.poisson_interval = 2.0 # opt.poisson_interval

    def DetectChangePoint(self, sequence, seq_no=0):
        event_time,_,intensity,intensity_times, change_points_true = sequence
        print('Change points true:',change_points_true)
        plt.plot(intensity_times, intensity)
        finish_time = event_time[-1]       
        event_count = DiscretizePoisson(event_time,self.poisson_interval) 
        # poisson_interval=1.0
        # data = np.array([])
        # for rate in [3,9,20]:
        #     new_data = random.poisson(lam=rate, size=300)
        #     data = np.append(data, new_data)
        # event_count = data
        # plt.plot(data)
        # plt.savefig('data.pdf')
        # plt.close()
        # sequence = Convert_to_Poisson(sequence) # TBD
        len_sequence = len(event_count)
        # print(gamma)
        lambdas = np.array([])
        posterior_runtime = np.array([])
        cp_scores = []
        r=0
        # exit()        
        for t,n in enumerate(event_count):
            # n=3

            # print('n',n)
            posterior_runtime = self.update_posterior_runtime(\
                posterior_runtime, lambdas, n)   
            # if t in range(300,350) or t in range(600,650):
            #     plt.plot(range(r,t+1),posterior_runtime)
            #     plt.savefig('v'+str(t)+'.png')
            #     plt.close()
            # print('v',posterior_runtime)
            # exit()
            lambdas = self.update_lambdas(lambdas, n) 
            # print('lambdas', lambdas)
            # exit()
            cp_index = r
            current_time = t * self.poisson_interval + self.poisson_interval/2.0
            cp_time = cp_index * self.poisson_interval + self.poisson_interval/2.0
            cp_score = posterior_runtime[0] if posterior_runtime.size > 0 else 1.0
            cp_scores += [(cp_index,current_time,\
                cp_score,cp_time)]
            # print('cp_time:',cp_time)
            
            if self.restart(posterior_runtime):
                # print('restart', 'True')
                r=t+1
                lambdas = np.array([])
                posterior_runtime=np.array([])
            # exit()
            # print('cp_scores',cp_scores)
            # print('t:',t,', cp_index:',r)
            # exit()
            # if t == 5:
            #     exit()
        # exit()
        # for i in range(100,106):
        #     print(cp_scores)
        cp_time_diff = [curr_time - cp_time for _,curr_time, _, cp_time in cp_scores]
        cp_time_diff = [x/50.0 for x in cp_time_diff]
        curr_time = [curr_time for _,curr_time, _,_ in cp_scores]
        plt.plot(curr_time, cp_time_diff)
        plt.savefig('Runtime'+str(seq_no)+'.png')
        plt.close()
        # exit()
        result, change_point_scores = self.Get_result(cp_scores, \
            finish_time, change_points_true, len_sequence) 
        # print('**'*50)
        return result, change_point_scores

    def restart(self,x):
        if x.size > 1 :
            if x[0] < np.max(x[1:]):
                return True 
        return False
    
        
    def update_lambdas(self, lambdas, n):
        lambdas[:] += n
        lambdas = np.append(lambdas,n) # Creating new Forecaster
        return lambdas

    def update_posterior_runtime(self, \
        v, lambdas, n):

        if v.size == 0:
            return np.array([1.0])
        else:
            t_r = lambdas.shape[0]
            # print('t-r', t_r)
            lambdas_scaled = np.divide( lambdas+1, \
                np.arange(len(lambdas))[::-1]+1) \
                if lambdas.size > 0 else np.array([1])
            

            likelihood = math.log(math.factorial(n)) \
                + lambdas_scaled \
                - n*np.log(lambdas_scaled)
            
            v_final = (1.0/(t_r+1))*math.exp(-np.sum(likelihood))
            v = (t_r/float(t_r+1))*np.exp(-likelihood)*v 
            
            # v_final = (1.0/300.0)*math.exp(-np.sum(likelihood))
            # v = (299/300)*np.exp(-likelihood)*v 
            
            v = np.append(v,v_final) 
            v = v/np.sum(v)         
            return v


    def Get_result(self,cp_scores, finish_time, change_points_true, num_intervals):
            
        cp_testing_times = [i_no*self.poisson_interval + \
            self.poisson_interval/2.0 for i_no in range(num_intervals)]
        
        cp_vector_true=GetCPVectorized(change_points_true, \
            [], cp_times=cp_testing_times) 
        
        cp_scores_dict=dict([(t,0.0) for t in cp_testing_times])
        for _,_,score,t in cp_scores:
            if t not in cp_scores_dict.keys():
                print('yes')
            cp_scores_dict[t]=max(score, cp_scores_dict[t])
        
        change_points_scores = [(t,cp_scores_dict[t]) \
            for t in cp_testing_times if cp_scores_dict[t] > 0 ]
        
        change_points_list = [(0,change_points_scores)]
        detection_delay=GetDetectionDelay(change_points_true, \
            change_points_list, finish_time)       
        # print(detection_delay)
        # exit()
        list_of_all_scores = [cp_scores_dict[t] for t in cp_testing_times]
        roc = roc_auc_score(cp_vector_true, list_of_all_scores)
        # print(roc)
        # exit()
        results={'roc':roc, 'change_points_estimated':change_points_scores, \
        'change_points_true':cp_vector_true,\
        'detection_delay':detection_delay}
        
        return results, change_points_scores
    
