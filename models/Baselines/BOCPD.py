from __future__ import division
import math
import numpy as np
import common.Utils as Utils


"""
--------------------------------------------------------------------------------------------------------------------------------------

Bayesian Online Change-point Detection original version

 Inputs:  
  -- environment: numpy array of piece-wise Bernoulli distributions

 Outputs:
  -- ChangePointEstimation: numpy array of change-point estimations

--------------------------------------------------------------------------------------------------------------------------------------
"""
class BOCPD:
    def __init__(self, opt=None):
        self.gamma = 1
        self.poisson_interval = 2.0 # opt.poisson_interval

    def DetectChangePoint(self, sequence):
        event_time,_,_,_, change_points_true = sequence
        print('Change points true:',change_points_true)

        finish_time = event_time[-1]       
        event_count = Utils.DiscretizePoisson(event_time,self.poisson_interval) 
        # poisson_interval=1.0
        # data = np.array([])
        # for rate in [3,6,9]:
        #     new_data = random.poisson(lam=rate, size=100)
        #     data = np.append(data, new_data)
        # sequence = data
        # sequence = Convert_to_Poisson(sequence) # TBD
        len_sequence = len(event_count)
        self.gamma = 1.0/len_sequence # Switching Rate 
        # print(gamma)
        lambdas = np.array([float(event_count[0])])
        _runtime = np.array([1.0])
        cp_scores = [(0,0,1, self.poisson_interval/2.0 )]
        #-----------------------------------------------------
        #Interation with the environment ...
        # print('Launching BOCD ...')
        # exit()
        
        for n in event_count[1:]:
            posterior_runtime = self.update_posterior_runtime(\
                posterior_runtime, lambdas, n, self.gamma)   
            # def int_to_str(x):
            #     return str(x)[:6]
            # strr = '  '.join(map(int_to_str,posterior_runtime))
            # print(strr)
            cp_index = np.argmax(posterior_runtime) 
            
            cp_time = cp_index * self.poisson_interval + self.poisson_interval/2.0
            # print('cp detected:',cp_time)

            cp_scores += [(\
                cp_index,cp_index,posterior_runtime[cp_index],\
                cp_time)]

            # print(cp_scores)

            lambdas = self.updatePoissonPrediction(lambdas, n) 
            # print(n)
            # print(lambdas)
        # exit()
        # for i in range(100,106):
        #     print(cp_scores)
        result, change_point_scores = self.Get_result(cp_scores, \
            finish_time, change_points_true, len_sequence) 
        
        return result, change_point_scores


    def Get_result(self,cp_scores, finish_time, change_points_true, num_intervals):
            
        cp_testing_times = [i_no*self.poisson_interval + \
            self.poisson_interval/2.0 for i_no in range(num_intervals)]
        # print(len(cp_testing_times))
        # exit()
        
        cp_vector_true=Utils.GetCPVectorized(change_points_true, \
            [], cp_times=cp_testing_times) 
        # print(cp_vector_true[140:150])
        # exit()

        cp_scores_dict=dict([(t,0.0) for t in cp_testing_times])
        for _,_,score,t in cp_scores:
            if t not in cp_scores_dict.keys():
                print('yes')
            cp_scores_dict[t]=max(score, cp_scores_dict[t])
        # for t in cp_testing_times[100:150]:
        #     print(cp_scores_dict[t])
        # exit()

        change_points_scores = [(t,cp_scores_dict[t]) \
            for t in cp_testing_times if cp_scores_dict[t] > 0 ]
        
        change_points_list = [(0,change_points_scores)]
        detection_delay=Utils.GetDetectionDelay(change_points_true, \
            change_points_list, finish_time)       
        # print(detection_delay)
        # exit()
        list_of_all_scores = [cp_scores_dict[t] for t in cp_testing_times]
        roc = Utils.roc_auc_score(cp_vector_true, list_of_all_scores)
        # print(roc)
        # exit()
        results={'roc':roc, 'change_points_estimated':change_points_scores, \
        'change_points_true':cp_vector_true,\
        'detection_delay':detection_delay}
        
        return results, change_points_scores
        
    def updatePoissonPrediction(self, lambdas, n):
        lambdas[:] += n
        lambdas = np.append(lambdas,n) # Creating new Forecaster
        return lambdas

    def update_posterior_runtime(self, \
        v, lambdas, n, gamma):

        lambdas_scaled = np.divide( lambdas+1, \
            np.arange(len(lambdas))[::-1]+1) \
            if lambdas.size > 0 else np.array([1])
        likelihood = np.exp(-lambdas_scaled - \
            math.log(math.factorial(n)) + n*np.log(lambdas_scaled))
        v_final = gamma*np.dot(likelihood, np.transpose(v)) 
        v = (1-gamma)*likelihood*v 
        v = np.append(v,v_final) 
        v = v/np.sum(v) 
        # print('max v', max(v))
        return v
