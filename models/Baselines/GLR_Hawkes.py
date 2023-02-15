import time
import sys
sys.path.append('../../common')
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
import torch
eps=sys.float_info.epsilon

class GLR_Hawkes:

    def __init__(self):
        self.beta=1
        self.L=30
        self.gamma=5
        self.MaxNumberIterationEM=20
        self.MinDiffAlpha = 0.0001
        self.step_of_thresholds=[]

    
    def DetectChangePoint(self, sequence):    
        LLRatio=self.ChangePointDetectionSequence(sequence) 
        # LLRatio=load_data(LLRatio_file)
        # LLRatio_plot(sequence, LLRatio)
        event_time,_,_,_, change_points_true = sequence
        finish_time = event_time[-1]
        cp_vector_true=GetCPVectorized(change_points_true, LLRatio)
        roc = ROC(LLRatio, cp_vector_true)
        change_points = ChangePointsFromLLRatio(LLRatio)
        detection_delay=GetDetectionDelay(change_points_true, \
            change_points, finish_time)       
        results={'roc':roc, 'change_points_estimated':change_points, \
        'change_points_true':cp_vector_true,\
        'detection_delay':detection_delay}
        return results, LLRatio
        
    def ChangePointDetectionSequence(self,sequence):
        event_time, event_type,_,_,_=sequence
        finish_time = event_time[-1]
        n_event=event_time.shape[0]
        print('finish time', finish_time)
        print('n event', n_event)
        LLRatio= []
        start_index = 0 
        end_index = GetEndIndex(event_time,event_time[0]+self.L)
        sub_event_time, sub_event_type = event_time[start_index:end_index],\
             event_type[start_index:end_index]
        _,_,mu,alpha_init = self.MLE_Estimation((sub_event_time, sub_event_type))
        set_of_alpha = [(start_index,end_index,alpha_init)]
        start_index = end_index
        while True:
            end_index = GetEndIndex(event_time,event_time[start_index]+self.L)
            sub_event_time, sub_event_type = \
             event_time[start_index:end_index],\
             event_type[start_index:end_index]

            alpha_last_window =set_of_alpha[-1][2]
            alpha = self.EstimateAlphaViaEM(mu, alpha_last_window, \
                sub_event_time, sub_event_type)
            set_of_alpha.append((start_index,end_index,alpha)) 
            alpha_init=GetAlphaLastPartition(set_of_alpha,start_index)
            LLRatio_window= self.GetLogLikelihoodRatio(mu, alpha_init, alpha,\
             sub_event_time, sub_event_type)
            LLRatio.append((start_index, end_index, LLRatio_window,event_time[start_index]))
            if event_time[start_index]+self.L > finish_time:
                break
            start_index += self.gamma
        return LLRatio


    def GetLogLikelihoodRatio(self,mu, alpha_init, alpha, event_time, event_type):

        intensities_alpha_init = GetIntensities(mu, alpha_init, self.beta, event_time)
        # print(intensities_alpha_init)
        intensities_alpha = GetIntensities(mu, alpha, self.beta, event_time)
        # for x in intensities_alpha:
        #     print(x)
        # # print(intensities_alpha[:20])
        # exit()
        finish_time = event_time[-1]
        event_time_gap = finish_time - event_time
        LLRatio = np.sum(np.log( intensities_alpha) - np.log(intensities_alpha_init))\
            - (alpha-alpha_init)*np.sum(1-np.exp(-self.beta*event_time_gap))
        return LLRatio

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
            

    def MLE_Estimation(self, sequence):
        event_time, event_type=sequence
        prediction_mse=0
        num_event = event_time.shape[0]
        # we assume f(x) = \sum_i log(x . c_i) - x . d
        finish_time=event_time[-1]
        d=np.array(
            [finish_time,\
             np.sum(1-np.exp( - self.beta*(finish_time-event_time)))])
        C = np.array([[1, x] for x in GetInfluences(event_time, self.beta)])
        mu,alpha,NLL=MLE_Estimate_param_SPG(C,d)
        return prediction_mse, NLL, mu, alpha # prediction mse TBD


    # exit()
    # threshold=5 # TBD get it from validation seq
    # average_NLL, average_prediction_mse = Prediction(change_points, sequence, threshold)
    # print(average_NLL)
    # print(average_prediction_mse)




  #     num_event=event_time.shape[0]
        # event_time=event_time.reshape(1, -1)
  #     event_time_col = event_time.reshape(-1,1)
  #     time_diff = np.repeat(event_time,num_event,axis=0) - np.repeat(event_time_col,num_event,axis=1)
  #     time_diff[np.triu(num_event,-1)]=0
  #     time_diff_decayed = alpha*self.beta*np.exp(-self.beta*time_diff)


    # def Prediction(self,change_points, set_of_sequences, threshold):
    #     for threshold_cp, change_points_threshold in change_points:
    #         if threshold_cp == threshold:
    #             selected_change_points = change_points_threshold
    #     total_NLL = 0
    #     total_prediction_mse = 0
    #     for change_points, sequence in zip(\
    #         selected_change_points, set_of_sequences):
    #         subsequences = split_the_data(change_points, sequence)
    #         NLL_sequence = 0 
    #         prediction_mse_sequence = 0
    #         for subsequence in subsequences:
    #             prediction_mse, NLL = self.MLE_Estimation(subsequence)
    #             NLL_sequence += NLL
    #             prediction_mse_sequence += prediction_mse
    #         total_NLL += NLL_sequence
    #         total_prediction_mse += prediction_mse_sequence
    #     total_number_events = get_total_number_of_events(set_of_sequences)
    #     average_NLL = 0
    #     average_prediction_mse = 0
    #     return average_NLL, average_prediction_mse