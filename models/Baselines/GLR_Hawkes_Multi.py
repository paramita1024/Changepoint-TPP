import time
import sys
sys.path.append('../data')
sys.path.append('../common')
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
from GLR_Hawkes_Multi_Classes import *
import torch
import Data_Generation_Module.Utils_data as Utils_d
eps=sys.float_info.epsilon

class GLR_Hawkes_Multi:

    def __init__(self, opt, d):
        
        # cp detection algorithm parameter
        self.L=10
        self.gamma=10
        # estimate A
        self.MaxNumberIterationEM=20
        self.MinDiffAlpha = 0.0001
        # tpp model parameter
        self.d=d
        self.beta=1
        # do not know
        self.step_of_thresholds=[]
        
    
    def DetectChangePoint(self, sequence, file_write):  

        def scale(x,y,z):
            x = np.array(x)
            x_min = x.min() 
            x_max = x.max()
            return y + x * ((z-y)/(x_max-x_min))

        
        LLRatio=self.ChangePointDetectionSequence(sequence) 
        LLRatio_times, LLRatio_values = \
            [x for _,_,_,x in LLRatio], [x for _,_,x,_ in LLRatio]
        # print(LLRatio)
        plt.plot(LLRatio_times, scale(LLRatio_values, 1.0,2.0), label='LLRatio')


        event_times, event_types, \
            merged_intensities, merged_intensity_times, cp \
                = sequence
        
        # print(np.max(merged_intensities[0]))
        # exit()

        file_write = file_write + '_multiple_node'
        # plt.plot(merged_intensity_times, merged_intensities, label='intensities')
        plt.scatter(cp, np.ones(len(cp)), \
			label='change-points', color='red',marker='x')
        plt.tight_layout()
        plt.legend()
        plt.savefig(file_write+'_Intensity.jpg')#'Conditional_Intensities.png'
        # Utils_d.plot_point_process(\
        #     event_times, merged_intensities, merged_intensity_times,\
        #     cp, path_write)

        exit()
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
        
        finish_time,n_event = event_time[-1], event_time.shape[0]
        LLRatio, start_index, end_index= [],0, GetEndIndex(event_time,event_time[0]+self.L)
        
        sub_event_time, sub_event_type = event_time[start_index:end_index],\
             event_type[start_index:end_index]
        
        # print(type(event_time))

        # print(type(event_type))
        # exit()

        event_obj = Events(self.d,  self.beta)        
        opt_ll = OptLogLikelihood( \
            sub_event_time, sub_event_type, \
            self.d, self.beta, event_obj)
        mu, A_init = opt_ll.optimize_X()
        

        # exit()
        set_of_A = [(start_index,end_index,A_init)]
        start_index = end_index
        
        Est_obj = EstimatorA(mu)
        llr_obj = LogLikelihoodratio(mu, self.d, self.beta)
        
        # exit()

        while True:

            
        
            end_index = GetEndIndex(event_time,event_time[start_index]+self.L)

            # print(start_index, end_index)
            

            # continue
        
            sub_event_time, sub_event_type = \
             event_time[start_index:end_index],\
             event_type[start_index:end_index]

            A_last_window =set_of_A[-1][2]

            init_data_structures = \
                event_obj.Init_data_structures( sub_event_time, sub_event_type)

            # exit()
            # continue
            # print(sub_event_type)
            A = Est_obj.Estimate_A(A_last_window, \
                init_data_structures)
            # exit()
            # print(A)
            # exit()
            set_of_A.append((start_index,end_index,A)) 
        
            # exit()
            # 
            A_init=GetAlphaLastPartition(set_of_A,start_index)
        
            LLRatio_window= llr_obj.LLR( A, A_init, init_data_structures)
            
            # exit()

            LLRatio.append((start_index, end_index,\
                 LLRatio_window,event_time[start_index]))
        

            if event_time[start_index]+self.L > finish_time:
                break
        
            start_index += self.gamma
        
        # exit()
            
        return LLRatio

