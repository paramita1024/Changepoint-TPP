import sys
sys.path.append('../common')
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
import torch
eps=sys.float_info.epsilon

class Score_statistics:

    def __init__(self, opt=None):
        self.beta=1
        self.L=10  #window length
        self.gamma=10    #next start index
        self.MaxNumberIterationEM=20
        self.MinDiffAlpha = 0.0001
        self.step_of_thresholds=[]
        self.opt=opt
        self.truncating_window=30   #interval of window

    def DetectChangePoint(self, sequence,llratio_plot):
        DiffGrad=self.ChangePointDetectionSequence(sequence) 
        # exit()
        # save('DiffGrad', DiffGrad)
        # exit()
        # DiffGrad=load_data('DiffGrad')
        LLRatio_plot(sequence, DiffGrad,llratio_plot)
        event_time,_,_,_, change_points_true = sequence
        finish_time = event_time[-1]
        cp_vector_true=GetCPVectorized(change_points_true, DiffGrad)
        roc = ROC(DiffGrad, cp_vector_true)
        # print(roc)
        # exit()
        change_points = ChangePointsFromLLRatio(DiffGrad)
        detection_delay=GetDetectionDelay(change_points_true, \
            change_points, finish_time)       
        results={'roc':roc, 'change_points_estimated':change_points, \
        'change_points_true':cp_vector_true,\
        'detection_delay':detection_delay}
        return results, DiffGrad

    def ChangePointDetectionSequence(self,sequence):
        event_time, event_type,_,_,_=sequence
        # exit()
        finish_time = event_time[-1]
        n_event=event_time.shape[0]
        DiffGrad= []
        start_index = 0 
        end_index = GetEndIndex(event_time,event_time[0]+self.L)
        # print(end_index)
        # exit()
        sub_event_time, sub_event_type = event_time[start_index:end_index],\
             event_type[start_index:end_index]
        mu,alpha_init = self.MLE_Estimation((sub_event_time, sub_event_type))
        start_index = end_index
        # exit()
        while True:
            cp_time=event_time[start_index]
            event_time_pre_cp=GetEventWindow(event_time, \
                start_index,self.truncating_window)
            end_index = GetEndIndex(event_time,cp_time+self.L)
            event_time_post_cp=GetEventWindow(event_time, end_index,self.truncating_window)
            # print( event_time_post_cp[0])
            # print(event_time_post_cp[-1])
            # exit()
            alpha_init=self.EstimateAlphaViaEM(mu, alpha_init, event_time_pre_cp, [])
            # print('alpha',alpha_init)
            # exit()
            grad_pre_cp=GetGradLikelihood(alpha_init, event_time_pre_cp, mu, self.beta)
            # exit()
            grad_post_cp=GetGradLikelihood(alpha_init, event_time_post_cp, mu, self.beta) 
            
            diff_grad=abs(grad_post_cp - grad_pre_cp)
            # print(diff_grad)

            DiffGrad.append((start_index, end_index, \
                diff_grad,cp_time))
            if finish_time - cp_time < self.L:
                break
            start_index += self.gamma
        # exit()
        return DiffGrad      

    def MLE_Estimation(self, sequence):
        event_time, event_type=sequence
        num_event = event_time.shape[0]
        # we assume f(x) = \sum_i log(x . c_i) - x . d
        finish_time=event_time[-1]
        d=np.array(
            [finish_time,\
             np.sum(1-np.exp( - self.beta*(finish_time-event_time)))])
        C = np.array([[1, x] for x in GetInfluences(event_time, self.beta)])
        mu,alpha,NLL=MLE_Estimate_param_SPG(C,d)
        return mu, alpha 



    def EstimateAlphaViaEM(self,mu, alpha_init, event_time, event_type):
        
        def CalculateP(alpha, event_time):
            num_event=event_time.shape[0]
            P = np.zeros((num_event, num_event))
            # print(P)
            arr = []
            old_t = event_time[0]
            for i,t in enumerate(event_time):
                if i>0:
                    diff_time = t-old_t
                    exp_del_t = math.exp(-self.beta*diff_time)
                    arr = [x*exp_del_t for x in arr] + \
                        [alpha*self.beta*exp_del_t]
                    old_t=t
                    sum_arr = sum(arr) 
                    # for 
                    P[:i,i]=np.array(arr).flatten()/(mu+sum_arr)
            return P

        def CalculateAlpha(P, event_time):
            final_time = event_time[-1]
            alpha = np.sum(P)/np.sum(1-np.exp(-self.beta*(final_time-event_time)))
            return alpha 

        # implemented for single dimension
        number_of_iter=0
        while True:
            P = CalculateP(alpha_init, event_time)
            alpha = CalculateAlpha(P, event_time)
            diff_alpha = (alpha-alpha_init)**2
            if diff_alpha < self.MinDiffAlpha \
                or number_of_iter > self.MaxNumberIterationEM:
                return alpha
            alpha_init=alpha
            number_of_iter += 1
            