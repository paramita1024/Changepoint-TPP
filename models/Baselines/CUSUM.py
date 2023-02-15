import random
import numpy.linalg as LA
import time
import sys
sys.path.append('../../common')
import numpy as np
import matplotlib.pyplot as plt
from Utils_baselines import *
from CUSUM_classes import *
import torch
eps=sys.float_info.epsilon
from numpy import random as np_random

class CUSUM:

    def __init__(self,opt):
        self.beta=1
        self.L=30
        self.gamma=5
        self.MaxNumberIterationEM=20
        self.MinDiffAlpha = 0.0001
        self.step_of_thresholds=[]
        self.max_index = 0
        self.threshold_ll = 0
        self.mu, self.A=0,0
        self.opt=opt
    
    def plot_point_process( self,event_times, intensity, time, cp, file_write, llr_dict):

        n_nodes = len(intensity)
        
        # figure, axis = plt.subplots(2,2)
        
        set_of_cp = event_times[np.where(cp==1)[0]]
        
        # print(set_of_cp)
        # max_t = max(event_times)
        # dt = 0.01
        
        # plt.clf()

        # for intensity_each  in intensity:

        # plt.clf()
        plt.plot(time, intensity[0], label='intensities')
        plt.scatter(set_of_cp, np.ones(set_of_cp.shape[0]), \
            label='change-points', color='red',marker='x')

        llr_dict_new =[]
        cp_list, val_list = [],[]
        i=0
        for key,value in llr_dict.items():
            # print(key)

            # print(value)

            # continue #exit()
            
            key_arr_list =[x for x in llr_dict[key].items()]
            
            # print( len(key_arr_list))

            # print( key_arr_list[:3])

            # exit()
            i+=1
            # if i == 6 :
            #     break

            # continue

            v  = [val for k, val in key_arr_list]
            k = [k for k, val in key_arr_list]
            

            cp = k[v.index(max(v))]
            

            # llr_dict_new += [(key*self.gamma, cp, max(v) )]
            cp_list += [event_times[cp]] 
            val_list += [max(v)]

        # exit()

        # print(cp_list)

        print(val_list)
        exit()
        plt.scatter(cp_list, val_list, label='llr values', color='purple')



        plt.title('Intensities')
        plt.tight_layout()

        plt.savefig('tmp.jpg')#file_write+'_Intensity.pdf')#'Conditional_Intensities.png')


        # exit()
        # return 

    def DetectChangePoint(self, sequence, i): 

        cusum_single = CUSUM_single(opt=self.opt, sequence=sequence) 
        # exit()
        llr_dict,estimated_cp = \
        cusum_single.CPDetectSequence()
        # exit()






        # print(llr_dict)
        # 
        # exit()
        if False:


            file_name = "../Results/hawkes_synthetic_s_1_nd_4_t_500_cp_3/CUSUM_Scores"
            llr_dict = load_data(file_name)[0]
            event_times,event_types,\
                merged_intensities,merged_intensity_times,\
                    cp = sequence

            self.plot_point_process(event_times, \
                merged_intensities, merged_intensity_times, cp, "", llr_dict)


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
        return [], llr_dict   #    results, LLRatio


class CUSUM_single:

    def __init__(self, opt, sequence):
        
        self.beta=1
        self.L=30
        self.B=50
        self.gamma=15
        self.MaxNumberIterationEM=20
        self.MinDiffAlpha = 0.0001
        self.step_of_thresholds=[]
        self.max_index = 0
        self.threshold_ll = 100
        self.mu, self.A=0,0
        self.d=opt.d

        self.event_times, self.event_types,_,_,_=sequence
        # print('**',type(event_times))
        # event_types = [random.randint(0,3) for _ in event_times]
        
        # self.event_times = event_times
        # self.event_types = event_types
        
        # n_event=event_times.shape[0]
        
    
    def set_cp(self, i):
        self.estimated_cp = i 
    
        
    def set_param(self,x=None):
        
        if x.any() == None:
            self.mu=np_random.rand(self.d)
            self.A=np_random.rand(self.d, self.d)
            for i in range(self.d): self.A[i]=1 
        else:
            self.mu = x[:,0].flatten()
            self.A = np.copy(x[:,1:])
        

    def get_events(self, start_time, end_time):

        indices = []

        for index, t in enumerate(self.event_times):
            
            if t > end_time:
                break

            if t >  start_time:
                indices += [index]

        # print(indices)
        if not indices:
            return -1,-1

        return indices[0], indices[-1]

    def CPDetectSequence(self):

        finish_time = self.event_times[-1]
        
        # exit()
        # exit()
        # exit()
        # print('yes')


        
        # train the model
        obj_ll = OptLogLikelihood(\
            self.event_times, self.event_types,B=self.B,d=self.d)
        # exit()

        self.set_param(obj_ll.X)
        
        
        n,max_ll,l=0,0,{0:{0:0}}
        
        self.set_cp(0)
        obj_llr = LogLikelihoodRatio(\
            self.event_times, self.event_types,\
            B=self.B, d=self.d,\
            param={'mu':self.mu,'A':self.A}\
            )
        # exit()

        # figure, axis = plt.subplot

        print(n)
        while True:

            n = n+1

            # print('N=',n)

            if n == 15:
                break

            # print('Iteration ',n)

            if (n)*self.gamma > finish_time:
                break

            # for i,t in enumerate(self.event_times[:20]): print(i,t)
            # start_index, end_index = self.get_events(\
            #     600,700)

            # exit()

            start_index, end_index = self.get_events(\
                (n-1)*self.gamma, n*self.gamma)


            # print('start_index:', start_index, ', end_index: ', end_index)

            # print('end_index: ', end_index)

            # start_index = 13 # delete
            # exit()   
            l[n]={}         
            for cp_index in range(start_index, (end_index+1)):
                l[n][cp_index]=obj_llr.GetLogLikelihoodRatio(\
                    cp_index, range(cp_index, (end_index+1)),\
                    end_time = n*self.gamma)

                # exit()

            # print(l[1])

            # exit()
            
            last_start_index, last_end_index = self.get_events(\
                (n-1)*self.gamma-self.B, (n-1)*self.gamma)


            # print('start_index:', last_start_index, ', end_index: ', last_end_index)
            

            if last_start_index >= 0:

                # print('True')
                for cp_index in \
                range(last_start_index, last_end_index+1):
                    
                    l[n][cp_index]=\
                    l[n-1][cp_index] + obj_llr.GetLogLikelihoodRatio(\
                        cp_index, range(start_index,(end_index+1)),\
                        start_time = (n-1)*self.gamma,\
                        end_time=n*self.gamma)
                #  + 
                    
            # exit()


            last_start_index, last_end_index = self.get_events(\
                (n-2)*self.gamma-self.B, (n-1)*self.gamma-self.B)


            # print('start_index:', last_start_index, ', end_index: ', last_end_index)
            
            
            # exit()
            if last_start_index >= 0:
                for index in \
                    range(last_start_index,(last_end_index+1)):
                    prev_max_ll = l[(n-1)][self.estimated_cp]    
                    if l[(n-1)][index] > prev_max_ll:
                        self.estimated_cp = index
            # print('yes')    
            # exit()
            

            l[n][self.estimated_cp] = \
                l[n-1][self.estimated_cp] + \
                obj_llr.GetLogLikelihoodRatio(\
                    self.estimated_cp,\
                    range(start_index,(end_index+1)),\
                    start_time = (n-1)*self.gamma,\
                    end_time=n*self.gamma)

            
            curr_cp_indices = list(l[n].keys())
            curr_LLR_list = [ l[n][key] for key in  curr_cp_indices]
            max_ll = max(curr_LLR_list)



            # if max_ll > self.threshold_ll:
            #     stopping_time = n*self.gamma 
            #     estimated_cp = curr_cp_indices[curr_LLR_list.index(max_ll)]
            #     self.set_cp(estimated_cp)
            #     break

            # print('LLR values')

            # for key in l.keys():

            #     print('Key: ', key, ' \n ', l[key])

            # exit()

            cp_indices = sorted([k for  k in l[n].keys()])
            llr_values = [l[n][k] for k in cp_indices]

            # print(llr_values)
            # plt.subplot(3,4,n)
            # plt.ylim([-100,100])
            plt.plot( cp_indices, llr_values)

            # print('yes')
        
        import time
        t = time.localtime()
        current_time = time.strftime("%H-%M-%S", t)
        # print(current_time)
        plt.savefig('tmp'+current_time+'.jpg')
        plt.clf()


        return l, self.estimated_cp