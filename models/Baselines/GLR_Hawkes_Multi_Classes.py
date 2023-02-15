import numpy as np  
import time
import sys
sys.path.append('../../common')
import matplotlib.pyplot as plt
from Utils_baselines import *
import numpy.random as np_random

class Events:

    def __init__(self, d, beta):

        self.d=d
        self.beta = beta


    def Init_data_structures( self, event_time, event_type):
            
        self.n = event_time.shape[0]

        # self.event_time = event_time 

        # self.event_type = event_type

        mask = np.zeros((self.n, self.d))
        mask[np.arange(self.n),event_type] = 1 
        mask = mask.T

        # print('n=', self.n, ',d=', self.d)
        # print('mask:',mask.shape)

        # exit()

        final_time = event_time[-1]
        time_decay = 1-np.exp(-self.beta*(final_time - event_time))

        # print(time_decay.shape)
        # plt.plot(time_decay)
        # plt.show()
        # exit()

        dt = event_time[:,None] - event_time[None,:]
        dt = self.beta*np.exp(-self.beta*dt)

        mask_lower_tri = np.tril(np.ones((self.n, self.n), dtype=int), -1)
        dt = np.multiply(mask_lower_tri, dt)
        
        influence = mask.dot(dt.T)
        # 
        # print(influence.shape)
        # plt.plot(influence[2,:])
        # plt.show()
        # exit()

        # print( np.sum(mask, axis=1))

        return (mask, time_decay, influence, dt)
    

class OptLogLikelihood:

    def __init__(self,event_times, event_types,d=None, beta=None, event_obj=None):
        
        
        self.d=d
        self.beta=beta
        self.n=event_times.shape[0]
        
        # self.num_seed=10 # no of times we run the algorithm
 
        # exit()
        
        self.mask, self.time_decay, self.influence, _ = \
            event_obj.Init_data_structures(event_times, event_types)
        self.influence = np.vstack((\
            np.ones((1,self.n)), self.influence\
        ))

        # print( np.sum(self.mask, axis=1))
        # exit()
        self.final_time=event_times[-1]
        
        # print(self.influence.shape)
        # plt.plot(self.influence[2])
        # plt.show()

        # exit()
        # exit()

        

    def optimize_X(self):

        X = np.zeros((self.d, self.d+1))

        # figure, axis = plt.subplots(self.d)

        for i in range(self.d):
            
            x,fval = self.optimize(self.mask[i], \
                self.influence, self.time_decay, self.final_time)
            X[i]=x

            # axis[i].plot(fval)
            # exit()
        # plt.show()

        # exit()
        
        return X[:,0].flatten(), X[:,1:]

    def optimize(self, \
        mask, influence, time_decay, final_time,\
        MAX_ITER_MLE=20, queue_length=100):
        
        # We are optimizing negative log likelihood of the given event sequence

        def f(x): 

            ITx=influence.T.dot(x)
            t1 = np.sum(np.multiply(mask, np.log(ITx)))

            z=np.sum( np.multiply(mask, time_decay))
            t2 = x.dot(\
                np.array(\
                [final_time] + \
                [z]*self.d)\
            )
            return t2-t1 

        def grad(x):

            inv_ITx = np.reciprocal(influence.T.dot(x))
            m_inv_ITx = np.multiply( mask, inv_ITx)
            reciprocal_term = influence.dot(m_inv_ITx)

            lin_term =np.array(\
                [final_time] + \
                [np.sum( np.multiply(mask, time_decay))]*self.d)
            return lin_term - reciprocal_term
            
        def proj(x):

            x[x<=0]=0.000001   

            return x

        spg_obj = spg()  

        x0  = np_random.rand(self.d+1)

        result= spg_obj.solve( x0, f, grad, proj, \
            queue_length , eps, MAX_ITER_MLE)

        # axis[math.floor(i/4), i%4].plot(result['buffer'])
        
        return result['bestX'], result['buffer']
        

class EstimatorA:

    def __init__(self, mu, MaxNumberIterationEM=20, MinDiffAlpha=0.0001):

        self.mu = mu 

        self.MaxNumberIterationEM = MaxNumberIterationEM

        self.MinDiffAlpha = MinDiffAlpha

    def update_lambda(self, A):

        self.lambda_t = np.sum(\
            np.multiply(\
            self.mu.reshape(-1,1) + A.dot(self.influence),\
            self.mask ), \
            axis=0).flatten()

        # print(self.lambda_t.shape)


    def CalculateP( self, A):
        
        
        P = np.multiply(\
            self.mask.T.dot( A ).dot(self.mask),\
            self.dt\
            )

        # print(P[3,4])

        # print(P[4,3])

        # P=np.arange(16).reshape(4,4)

        # self.lambda_t = np.array([1,2,3,4])

        # print(P)

        P = np.divide( P , \
            self.lambda_t[:,None]
            )

        # print(P) 

        # exit()

        # print(P[3,4], P[4,3])

        # P = np.multiply(P, \
        #     np.tril(np.ones((self.n, self.n), dtype=int), -1))

        # print(P[3,4], P[4,3])

        return P  

    def CalculateA(self,P):

        divisor = self.mask.dot(self.time_decay)
        #print(self.time_decay)
        #print(self.mask)
        #print(divisor)
        #print(divisor.shape)
        #print(self.time_decay.shape)
        #print(self.mask.shape)
        #exit()
        divisor[divisor<=0] = 0.000001
        
        return np.divide(\
            self.mask.dot(P).dot(self.mask.T),\
            divisor[None,:]\
            )

    def set_variables(self, A_init, init_data_structures):

        self.mask, self.time_decay, self.influence, self.dt = init_data_structures

        self.d, self.n = self.mask.shape

        self.update_lambda(A_init)


    def Estimate_A( self, A_init, init_data_structures):

        number_of_iter=0

        self.set_variables( A_init, init_data_structures)

        while True:
            
            P = self.CalculateP(A_init)

            # exit()
            
            A = self.CalculateA(P)

            # exit()

            self.update_lambda(A)
            
            diff_alpha = LA.norm(A - A_init,'fro')
            
            if diff_alpha < self.MinDiffAlpha \
                or number_of_iter > self.MaxNumberIterationEM:
            
                return   A 
            
            A_init= A 
            number_of_iter += 1




class LogLikelihoodratio:

    def __init__(self, mu, d, beta):

        self.mu = mu 
        self.d=d
        self.beta = beta


    def LLR(self, A, A_prev, init_data_structures):

        mask, time_decay, influence, _ = init_data_structures

        # exit()

        lambda_new_param = self.mu.reshape(-1,1) + A.dot(influence)

        lambda_old_param = self.mu.reshape(-1,1) + A_prev.dot(influence)

        log_val =  np.sum(\
            np.multiply( \
            mask,\
            np.log( np.divide(lambda_new_param, lambda_old_param) ) \
             )\
        ) 
        linear_val = (mask.dot(time_decay)).dot(\
            np.sum(A-A_prev, axis=0)\
        )
        return log_val - linear_val 
