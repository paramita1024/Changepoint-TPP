import math
import random
import numpy.linalg as LA
import time
import sys
sys.path.append('../../common')
import numpy as np
import numpy.random as np_random
import matplotlib.pyplot as plt
from Utils_baselines import *
import torch

eps=sys.float_info.epsilon


class Events:

    def __init__(self, event_times, event_types, B=None, d=None):

        self.event_times = event_times 
        self.event_types = event_types
        self.B=B  
        self.d=d
        self.list_of_C=[] 
        self.D=[]

        self.summarize()

    def get_events(self, start_time, end_time):

        event_times, event_types = [],[]
        
        for tm,tp in zip(self.event_times, self.event_types):
            
            if tm > end_time:
                break

            if tm >  start_time:
                event_times += [tm]
                event_types += [tp]
        
        return event_times, event_types


    def summarize(self):

        final_time = self.event_times[-1]

        influence, num_events_type = [0]*self.d,[0]*self.d  
        event_times_decayed = \
            np.exp(-(final_time - np.array(self.event_times))  )
        # exit()
        
        for event_time_decay,event_type in \
            zip(event_times_decayed, self.event_types):
            
            influence[event_type] += event_time_decay
            num_events_type[event_type]+=1 

        # print(num_events_type)

        Z=[n-i for n,i in zip(num_events_type,influence)]
        self.D = np.array([ [final_time] + [z]*self.d  for z in Z])

        tmp_list_of_C = [[] for i in range(self.d)]
        for event_time, event_type in \
            zip(self.event_times, self.event_types):
            
            final_time = event_time

            sub_event_times, sub_event_types = self.get_events(\
                event_time-self.B, event_time )
            
            sub_event_times = np.exp(-(event_time - \
                np.array(sub_event_times)))
            
            influence_C= [0]*self.d
            
            for event_time_h, event_type_h in \
                zip(sub_event_times, sub_event_types):
                influence_C[event_type_h] += event_time_h

            tmp_list_of_C[event_type]  += [[1] + influence_C]

        self.list_of_C = [np.array(C) for C in tmp_list_of_C]        
        
class OptLogLikelihood:

    def __init__(self,event_times, event_types,B=None,d=None):
        
        self.event_times=event_times
        self.event_types=event_types
        self.d=d
        self.num_seed=10 # no of times we run the algorithm
 
        event_obj = Events(self.event_times, self.event_types,B=B, d=self.d)
        # exit()

        # print(self.d)
 
        self.D =event_obj.D
        self.list_of_C=event_obj.list_of_C
        
        # print( self.D.shape)

        # for c in self.list_of_C:
        #     print(c.shape)

        # exit()

        self.X = self.optimize()

    
    def optimize(self):

        # figure, axis = plt.subplots(4)

        X=[]
        for i,C,d in zip(range(self.d),self.list_of_C,self.D):

            x,f_x = self.MLE_Estimate_param_SPG(C,d, \
            MAX_ITER_MLE=20, queue_length=100)

            X += [x] 
            # axis[i].plot(f_x)


        # plt.savefig('tmp.jpg')

        # X = np.array(X)
        # print(X.shape)
        # exit()
        return np.array(X)



    def MLE_Estimate_param_SPG(self, C,d, MAX_ITER_MLE=20, queue_length=100):
        
        def f(x):
            return -(np.sum(np.log(C.dot(x)))-d.dot(x))
        def grad(x):
            z=1.0/(C.dot(x))
            return d - C.T.dot(z).flatten()
        def proj(x):
            x[x<0]=0
            return x

        
        spg_obj = spg()  

        NLL_list, param_list, buffer=[],[],[]

        # figure, axis = plt.subplots(3,4)

        for i in range(self.num_seed):

            x0  = np_random.rand(self.d+1)

            # print(x0)
            
            # exit()
            result= spg_obj.solve( x0, f, grad, proj, \
                queue_length , eps, MAX_ITER_MLE)
    
            # axis[math.floor(i/4), i%4].plot(result['buffer'])
            
            # exit()

            param_list.append(result['bestX'])
            NLL_list.append(f(result['bestX']))
            buffer.append(result['buffer'])
        
        # plt.savefig('tmp.jpg')
        # exit()
        index_for_minimum_NLL = np.argmin(np.array(NLL_list))
        
        return param_list[index_for_minimum_NLL], buffer[index_for_minimum_NLL]
        




class LogLikelihoodRatio:

    def __init__(self, event_times, event_types, \
        d=None, B=None, \
        param=None):
        
        self.event_times=event_times
        self.event_types=event_types
        self.d=d 
        self.B=B
        self.mu=param['mu']
        self.A=param['A']
        # self.A_after =  np_random.rand(self.d, self.d)

    def get_events(self, start_time, end_time):

        event_times, event_types = [],[]
        
        for tm,tp in zip(self.event_times, self.event_types):
            
            if tm > end_time:
                break

            if tm >  start_time:
                event_times += [tm]
                event_types += [tp]
        
        return event_times, event_types

    def EventsInfluence(self, index, flag_CP=False, cp_index=None):
        
        curr_time = self.event_times[index]
        start_time = max(curr_time - self.B,0)

        if flag_CP:
            cp_time = self.event_times[cp_index]
            if start_time < cp_time:
                start_time = cp_time
        else:
            curr_time = self.event_times[index]
            start_time = curr_time - self.B

        event_times, event_types = \
            self.get_events(start_time, curr_time) 

        event_times = np.array(event_times)
        events_decayed = np.exp(-(curr_time - event_times))

        events_influence = np.zeros(self.d)
        for event, type_event in zip(events_decayed, event_types):
            events_influence[type_event] += event

        return events_influence 

    def GetIntensities(self, subset_of_interest, flag_CP=None, cp_index=None):
        
        # print(len(subset_of_interest))

        events_influence = np.array(\
            [self.EventsInfluence(\
                index, flag_CP=flag_CP, cp_index=cp_index) \
            for index in subset_of_interest]).T
        
        # print(events_influence.shape)
        # print(np.tile(self.mu.reshape(1,-1).T,(1,len(subset_of_interest))))
        # exit()

        # if flag_CP:
            
        #     A = self.A_after

        # else:

        #     A = self.A   

        Lambda = np.tile(\
            self.mu.reshape(-1,1),\
            (1,len(subset_of_interest) ) )\
                 + self.A.dot(events_influence)

        # print(Lambda.shape)
        return Lambda

    def GetLogLikelihoodRatio(self, cp_index, subset_of_interest,\
        start_time=None, end_time=None): 
        

        # print( 'cp_index' , cp_index)

        # print('subset_of_interest', subset_of_interest)

        # exit()
        # exit()
        # cp_index = 10
        # subset_of_interest = range(20,50)
        lambda_cp=self.GetIntensities(subset_of_interest, \
            flag_CP=True, cp_index=cp_index)
        # exit()
        
        # print(lambda_cp.shape)

        # print(lambda_cp[:,-1].reshape(-1,1).shape)
        # exit()

        
        # print(lambda_cp.shape)

        # exit()
        lambda_inf = self.GetIntensities(subset_of_interest, \
            flag_CP=False)
        # exit()
        # print(len(subset_of_interest))
        # exit()
        event_times = self.event_times.copy()[subset_of_interest]

        # print('lambda_cp')

        # plt.plot( lambda_cp, label='lambda cp')


        # plt.savefig('tmp cp.jpg')

        # # exit()

        # plt.clf()

        # plt.plot( lambda_inf, label='lambda inf')
        
        # plt.savefig('tmp inf.jpg')

        # exit()

        if start_time and end_time:
            # print('**')
            # exit()
            
            event_times = np.hstack((start_time, event_times))
            event_times = np.hstack((event_times, end_time))

            # print(event_times.shape)
            # exit()
            # print('l b4',lambda_cp.shape)
            lambda_cp = np.hstack((\
                lambda_cp,\
                lambda_cp[:,-1].reshape(-1,1)))
            # print('l after',lambda_cp.shape)
            # exit()
            lambda_inf= np.hstack((\
                lambda_inf,\
                lambda_inf[:,-1].reshape(-1,1)))
        
        # print(event_times.shape)
        if not start_time and end_time:
            # print('True')
            # print(type(event_times))
            event_times = np.hstack((event_times, \
                end_time))
            # print('*',event_times.shape)

            # exit()

        # print(event_times.shape)
        del_t = event_times[1:]-event_times[:-1]
        
        # print('x',lambda_inf.shape)

        # print('xp',lambda_cp.shape)
        
        # print('t',event_times.shape)

        # print('d',del_t.shape)
        # exit()

        # tmp = np.sum(lambda_cp-lambda_inf, axis=0)
        #.dot(del_t)

        # print(tmp.shape)
        # def my_log_division(X,Y):
        #     X[X==0]=0.0001
        #     Y[Y==0]=0.0001
            
        #     Z = np.divide(X,Y)
        #     # Z[Y==0]=0
        #     Z_log=np.log(Z)
        #     # Z_log[Z==0]=0
        #     return Z

        # def my_log_division_2(X,Y):
        #     X[X==0]=0
        #     Y[Y==0]=0
            
        #     Z = np.divide(X,Y)
        #     Z[Y==0]=0
        #     Z_log=np.log(Z)
        #     Z_log[Z==0]=0
        #     return Z

        # additional term to overcome computational error

        lambda_cp[lambda_cp==0] = 0.00001
        lambda_inf[lambda_inf==0] = 0.00001

        # z_1 = np.sum(np.log(np.divide(lambda_cp, lambda_inf)))

        # print(z_1)

        # exit()
# 
        # z_2 =     np.sum(lambda_cp-lambda_inf, axis=0).dot(del_t)

        # print('z1',z_1)
        # print('z2', z_2)

        # exit()

        return np.sum(np.log(np.multiply(mask,np.divide(lambda_cp, lambda_inf)))) - \
            np.sum(lambda_cp-lambda_inf, axis=0).dot(del_t)

        # print( x )
        # return x 