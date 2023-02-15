import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import numpy.random as np_random
import random
import math
import numpy.linalg as LA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sys
import shutil
import os

eps=sys.float_info.epsilon

def save_model(model, opt, dir, init=False, intermediate=False, epoch=0, final=False):
    if init==True:
        file_name = "Init_model_" + str(opt.data) + "_learning_rate_" + str(opt.learning_rate) + "_num_changepoints_" + \
            str(opt.num_changepoints) + "_device_" + str(opt.device) + "_method_" + str(opt.method) + \
                "_event_embedding_" + str(opt.d_model) + "_transformer_hid_dim_" + str(opt.d_hid) + \
                    "_n_layers_" + str(opt.n_layers) + "_n_head_" + str(opt.n_head) 

    if intermediate == True:
        file_name = "Intermediate_model_" + str(opt.data) +"_epoch_"+ str(epoch) + "_learning_rate_" + str(opt.learning_rate) + "_num_changepoints_" + \
            str(opt.num_changepoints) + "_device_" + str(opt.device) + "_method_" + str(opt.method) + \
                "_event_embedding_" + str(opt.d_model) + "_transformer_hid_dim_" + str(opt.d_hid) + \
                    "_n_layers_" + str(opt.n_layers) + "_n_head_" + str(opt.n_head)
    
    if final == True:
        file_name = "Final_model_" + str(opt.data) + "_learning_rate_" + str(opt.learning_rate) + "_num_changepoints_" + \
            str(opt.num_changepoints) + "_device_" + str(opt.device) + "_method_" + str(opt.method) + \
                "_event_embedding_" + str(opt.d_model) + "_transformer_hid_dim_" + str(opt.d_hid) + \
                    "_n_layers_" + str(opt.n_layers) + "_n_head_" + str(opt.n_head)

    torch.save(model,dir+file_name)

def compute_time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se

def get_gradient(data = None, optimizer=None, model=None, num_types=1):

    event_time, event_type, event_feat = data
    _, _, prediction = model.forward(event_time, event_type, event_feat)
    lambdas , time_prediction, type_prediction = prediction

    num_seq, num_events=event_time.size()
    mask=torch.zeros(num_seq, num_events, num_types)
    mask[:,torch.arange(num_events),  event_type -1]=1

    nll_loss = log_likelihood_array(lambdas.squeeze(), event_time, mask)

    optimizer.zero_grad()
    nll_loss.backward()
    
    params = list( model.linear.parameters())

    param_grad = torch.Tensor([])

    for param in params:
        x = param.grad.data.detach().flatten()
        param_grad = torch.hstack((param_grad,x))
    
    return param_grad, nll_loss, prediction

def Get_Score_Stat(dlt_dA, dlt_w_dA, t, w):

    n=dlt_dA.shape[0]
    dlt_dA = dlt_dA.reshape(-1,1)
    dlt_w_dA = dlt_w_dA.reshape(-1,1)
    lambdas=1.0

    dlt_dA_diff = dlt_dA - dlt_w_dA
    
    inv_fisher_mat = (1.0/t)*LA.inv(lambdas*torch.eye(n) + dlt_dA.mm(dlt_dA.T))
    x = inv_fisher_mat.mm(dlt_dA_diff)
    ss_score = (1.0/w)*dlt_dA_diff.T.mm(x)

    ss_score = ss_score.flatten()[0]

    return ss_score

def compute_integral(all_lambda, time):
    """ Log-likelihood of non-events, using linear interpolation. """
    diff_time = (time[:, 1:] - time[:, :-1])
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]).squeeze(-1)
    biased_integral =  diff_lambda * diff_time
    result = 0.5 * biased_integral
    result = torch.sum(result, dim=-1)
    return result

def log_likelihood(lamda, time):
    """ Log-likelihood of sequence. """
    return -(torch.sum(torch.log(lamda)) - compute_integral(lamda, time))

def compute_integral_all(lamda, time):
    diff_time = (time[:,1:] - time[:,:-1]).squeeze()
    diff_lamda = (lamda[1:] + lamda[:-1]).squeeze()
    
    return diff_time*diff_lamda

def log_likelihood_array(lamda, time, mask=None):
    if mask is not None:
        log_term = torch.sum(mask * torch.log(lamda)) 
        lin_term =  torch.sum(compute_integral_all(torch.sum(lamda, dim=-1), time))
        return - (log_term-lin_term)
    else:
        return -(torch.log(lamda[1:]) - compute_integral_all(lamda, time))

def log_likelihood_mark_array(mark,target):
    return F.nll_loss(mark, target, reduction='none')

def log_ratios(data, lambdas, device):
    event_time = torch.unsqueeze(data, dim=0)
    diff_time = data[1:] - data[:-1]

    num_changes = lambdas.shape[0]-1
    N = diff_time.shape[0]
    arr = torch.zeros(num_changes,N).to(device)
    for i in range(num_changes):
        arr[i] = log_likelihood_array(lambdas[i], event_time) - log_likelihood_array(lambdas[i+1], event_time)
    return arr

def MakeCopy(FromA,ToB):
    for a in FromA.__dict__.keys():
        # print(a)
        ToB.__dict__[a] = FromA.__dict__[a]

def save(path_write, x):
    with open(path_write+'.pkl','wb') as f:
        pickle.dump( x, f,  protocol=pickle.HIGHEST_PROTOCOL)

def create_dir(dir=None):
    if dir is not None:
        shutil.rmtree(dir, ignore_errors=True)
        try:
            os.makedirs(dir)
        except:
            pass

def load_data(name):
    with open(name+'.pkl', 'rb') as f:
        data = pickle.load(f)#, encoding='latin-1')
        # num_types = data['dim_process']
        # print(dict_name, len(data[dict_name]))
        # data = data#[dict_name]
        # print('Number of sequences', len(data))
        return data#, int(num_types)

def load_dataset(folder):
    data_init = load_data(folder)
    num_sequences = len(data_init['time'])
    dataset = []
    for i in range(num_sequences): 
        dataset += [\
        (data_init['time'][i],\
        data_init['mark'][i],\
        data_init["intensities"][i], \
        data_init["intensity_times"][i], \
        data_init['change_points'][i], \
        data_init['features'][i], \
        data_init['num_types'][i], \
        data_init['len_seq'][i], \
        data_init['len_feat'][i], \
        data_init['run_times'][i])]
    return dataset

def Plot_time_count(intensity_times, intensity, event_count, save_file):
    # plt.plot(event_time,len(event_time)*[1],label='event time', \
    #     marker='*',markersize=.2,linewidth=.50 )
    plt.plot(intensity_times, intensity,label='intensity')
    plt.plot([i*2 for i in range(len(event_count))], \
        [i/2.0 for i in event_count], label='event count')
    plt.legend()
    plt.savefig(save_file)

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def split_data_GLR(data, start_index, cp_index, end_index):

    event_times, event_types, event_features = data 

    # print(event_times.size(), event_types.size(), event_features.size())

    # exit()

    data_before_cp = ( event_times[:,start_index:cp_index], \
        event_types[:,start_index:cp_index], event_features[:,start_index:cp_index,:])

    data_after_cp = ( event_times[:,cp_index:end_index], \
        event_types[:,cp_index:end_index], event_features[:,cp_index:end_index,:])

    return data_before_cp, data_after_cp

def split_data_SS(data, cp_index, end_index):

    event_times, event_types, event_features = data 

    data_before_cp = ( event_times[:,:cp_index], \
        event_types[:,:cp_index], event_features[:,:cp_index,:])

    data_after_cp = ( event_times[:,:end_index], \
        event_types[:,:end_index], event_features[:,:end_index,:])

    return data_before_cp, data_after_cp

def GetDetectionDelay(cp_true, cp_estimated_list, finish_time):
    
    def find_elm(change_points, cp_true, finish_time):
        for cp,_ in change_points:
            # print(cp)
            # print(cp_true)
            # exit()
            if cp>cp_true:
                return cp
        return finish_time
    
    avg_detection_delay = []
    # print(cp_estimated_list)
    for threshold, cp_threshold in cp_estimated_list:
        # print(threshold)
        # print('cp_threshold', cp_threshold)
        # print('cp_true', cp_true)
        # time.sleep(2)
        detection_delay = []
        # print(cp_true)
        # exit()
        for i,cp in enumerate(cp_true):
            # print(cp_threshold)
            # print(cp)
            # print(finish_time)
            # exit()
            cp_estimated = find_elm(cp_threshold, cp, finish_time)
            # print(cp, cp_estimated)
            # time.sleep(2)
            detection_delay += [cp_estimated-cp]
        # exit()
        avg_detection_delay += [(threshold, \
            float(sum(detection_delay))/len(detection_delay))]
        # print(avg_detection_delay[-1])
    return avg_detection_delay

def ChangePointsFromLLRatio(LLRatio):
    LLRatio_values= np.array([x[2] for x in LLRatio if x[2] > 0 ])
    min_positive_LLRatio = np.min(LLRatio_values)
    sorted_threshold_values=np.sort(LLRatio_values)+min_positive_LLRatio/2
    set_of_threshold_values = np.pad( sorted_threshold_values, (1,0),'constant')
    change_points=[\
        (threshold, [ (cp_time, LLRatio_window)  for \
        start_index, end_index, LLRatio_window, cp_time  in LLRatio if LLRatio_window > threshold]) \
                for threshold  in set_of_threshold_values]
    return change_points

def ROC(LLRatio, change_points_true):
    LLRatio_values= np.array([x[2] for x in LLRatio if x[2] > 0 ])
    norm_val = math.sqrt(np.sum(np.array(LLRatio_values)**2))
    normalized_LLRatio_values = [x/norm_val for x in LLRatio_values]
    roc=roc_auc_score(change_points_true, normalized_LLRatio_values)
    return roc

def GetCPVectorized( list_of_change_points, LLRatio, cp_times=None):
    if cp_times is None:
        cp_times = [x[3] for x in LLRatio if x[2]>0]
    cp_indicator, i=[],0
    for t in cp_times:
        if i < len(list_of_change_points):
            if t>=list_of_change_points[i]:
                cp_indicator += [1]
                i += 1
            else:
                cp_indicator += [0]     
        else:
            cp_indicator += [0]
    return cp_indicator

def cp_from_scores(scores_times, scores_values, num_change_points):
    idx = -1*num_change_points
    top_cp_idx = np.argsort(scores_values)[idx:]
    cp_estimates = [scores_times[i] for i in top_cp_idx]
    return cp_estimates

def interarrival_cp_plot(event_times, cp_true, cp_estimates, save_dir, file_name="inter_arrival"):
    plt.plot(event_times[1:],event_times[1:]-event_times[:-1],label="Inter arrival time")
    plt.scatter(cp_true,np.ones_like(cp_true),label="actual change_points",c='red')
    plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()

def intensity_cp_plot(intensities, intensity_times, cp_true, cp_estimates, save_dir, file_name="intensity"):
    plt.plot(intensity_times, intensities,label='True intensity')
    plt.scatter(cp_true,np.ones_like(cp_true),label="true change points",c='red')
    plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()

def LLRatio_plot(dataset, LLRatio, save_dir, file_name="LLRatio"):

    time_indices = [x[3] for  x in LLRatio]
    LLRatio_list=[x[2] for x in LLRatio]
    cp_true = dataset[4]
    event_times = dataset[0]
    cp_estimates = cp_from_scores(time_indices, LLRatio_list, len(cp_true)) 

    plt.plot(time_indices, LLRatio_list,label='LLRatio')
    plt.scatter(cp_true,np.ones_like(cp_true),label="true change points",c='red')
    plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()

def GetIntensities(mu, alpha, beta, event_time):
    intensity=mu
    intensities=[intensity] 
    old_t=event_time[0]
    for event_t in event_time[1:]:
        diff_t=event_t-old_t
        intensity=mu+(intensity-mu+alpha*beta)*math.exp(-beta*diff_t)
        old_t=event_t 
        intensities.append(intensity)
    return np.array(intensities)

def GetInfluences(event_time, beta):
    influence=0
    influences=[influence]
    old_t=event_time[0]
    for event_t in event_time[1:]:
        diff_t=event_t-old_t
        influence_new = (influence+beta)*math.exp(-beta*diff_t)
        old_t=event_t 
        influences.append(influence_new)
        influence=np.copy(influence_new)
    return np.array(influences)

def split_the_data(change_points, sequence):
    splits = []
    last_change_point=0
    for change_point in change_points:
        splits += sequence[last_change_point:change_point]
        last_change_point = change_point
    splits += sequence[last_change_point:]
    return splits

def get_total_number_of_events(set_of_sequences):
    count = 0 
    for sequence in set_of_sequences:
        event_time, event_type = sequence
        count += event_time.shape[0]
    return count 

def GetEndIndex(event_time, finish_time):
    n_event = event_time.shape[0]
    for i,t in enumerate(event_time[::-1]):
        if t <= finish_time:
            return n_event - i


def DiscretizePoisson(event_time, interval):
    event_count=[]
    finish_time=event_time[-1]
    i,start_time=0,event_time[0]
    while True:
        end_time = start_time + interval
        count=0
        while event_time[i] < end_time:
            count += 1 
            i+=1
        event_count += [count]
        start_time = end_time
        if start_time+interval>finish_time:
            break
    return event_count

def GetEventWindow(event_time, finish_index, interval):
    n_events=len(event_time)
    if finish_index < len(event_time):
        finish_time = event_time[finish_index]
    else:
        finish_time=event_time[-1]
    start_time = finish_time - interval
    if start_time < event_time[0]:
        return event_time[:finish_index]
    for i,t in enumerate((event_time)):
        # print(t,n_events-i-1)
        # time.sleep(3)
        # print
        if t > start_time:
            start_index = i
            # print('start index', start_index)
            # print('end index',finish_index)
            return event_time[start_index:finish_index]
    
def GetGradLikelihood(opt, event_times, mu, alpha):
    n=event_times.shape[0]
    influence=np.zeros(n)
    final_time=event_times[-1] # define
    for i in range(n):
        t=event_times[i]

        indices = np.logical_and(\
            t-opt.truncating_window<event_times,
            event_times<t)
        dt_vec=t-event_times[indices]
        influence[i] =\
            opt.beta*np.sum(np.exp(-opt.beta*dt_vec))
    ll = np.sum(np.log(mu + alpha*influence)) \
        - mu*final_time \
        - alpha*(1-(influence[-1]/opt.beta))

    grad = np.reciprocal(mu+alpha*influence).dot(influence) - (1-(influence[-1]/opt.beta))
    return grad 

def MLE_Estimation(opt, sequence):
    event_time, event_type=sequence
    num_event = event_time.shape[0]
    # we assume f(x) = \sum_i log(x . c_i) - x . d
    finish_time=event_time[-1]
    d=np.array(
        [finish_time,\
            np.sum(1-np.exp( - opt.beta*(finish_time-event_time)))])
    C = np.array([[1, x] for x in GetInfluences(event_time, opt.beta)])
    mu,alpha,NLL=MLE_Estimate_param_SPG(C,d)
    return mu, alpha

def EstimateAlphaViaEM(opt, mu, alpha_init, event_time, event_type):
    
    def CalculateP(alpha, event_time):
        num_event=event_time.shape[0]
        P = np.zeros((num_event, num_event))
        # print(P)
        arr = []
        old_t = event_time[0]
        for i,t in enumerate(event_time):
            if i>0:
                diff_time = t-old_t
                exp_del_t = math.exp(-opt.beta*diff_time)
                arr = [x*exp_del_t for x in arr] + \
                    [alpha*opt.beta*exp_del_t]
                old_t=t
                sum_arr = sum(arr) 
                # for 
                P[:i,i]=np.array(arr).flatten()/(mu+sum_arr)
        return P

    def CalculateAlpha(P, event_time):
        final_time = event_time[-1]
        alpha = np.sum(P)/np.sum(1-np.exp(-opt.beta*(final_time-event_time)))
        return alpha 

    # implemented for single dimension
    number_of_iter=0
    while True:
        P = CalculateP(alpha_init, event_time)
        alpha = CalculateAlpha(P, event_time)
        diff_alpha = (alpha-alpha_init)**2
        if diff_alpha < opt.MinDiffAlpha \
            or number_of_iter > opt.MaxNumberIterationEM:
            return alpha
        alpha_init=alpha
        number_of_iter += 1


def GetAlphaLastPartition(set_of_alpha,start_index):
    
    for i,(start, end, alpha) in enumerate(set_of_alpha):
        if end > start_index:
            print('Shift initial window. Not getting alpha from prior segments')
        if end <= start_index:
            next_end = set_of_alpha[i+1][1]
            if next_end  > start_index:
                # print('start, end, alpha', start, end, alpha)
                return alpha
    
def proj(x):
    x[x<0]=0
    return x

def MLE_Estimate_param_SPG(C,d, MAX_ITER_MLE=20, queue_length=100):
    def f(x):
        return -(np.sum(np.log(C.dot(x)))-d.dot(x))
    def grad(x):
        z = 1.0/(C.dot(x))
        return d - C.T.dot(z)
    
    spg_obj = spg()  

    range_of_mu=np.random.uniform(0,1,20)
    range_of_alpha=np.random.uniform(0,1,20)
    NLL_list=[]
    param_list=[]
    for mu,alpha in zip(range_of_mu, range_of_alpha):

        x0  = np.array([mu,alpha])#np.ones(2)
        result= spg_obj.solve( x0, f, grad, proj, \
            queue_length , eps, MAX_ITER_MLE)
        param_list.append(result['bestX'])
        NLL_list.append(f(result['bestX']))
        # print('*'*50)
    # print(NLL_list)
    index_for_minimum_NLL = np.argmin(np.array(NLL_list))
    mu,alpha = param_list[index_for_minimum_NLL]
    NLL=NLL_list[index_for_minimum_NLL]
    return mu, alpha, NLL
   
def MLE_Estimate_param_GD(C,d):
    # we assume f(x) = \sum_i log(x . c_i) - x . d
    x=np.array([1,1])#[np.random.rand(1).flatten(),np.random.rand(1).flatten()] )# rand # mu_alpha
    # print(x)
    # exit()
    n=1
    while True:
        grad = C.T.dot(C.dot(x))-d 
        # print('grad norm', LA.norm(grad))
        if LA.norm(grad) < eps:
            break
        learning_rate=0.0001*(1.0/n)
        x = x + learning_rate*grad
        x = proj(x)
        LL = np.sum(np.log(C.dot(x)))-d.dot(x)
        print('LL', LL, ' x',x)
        n+=1
        if n == 20:
            break
    return x[0],x[1]
        
def find_indicator(x,y):
    j,z=0,[]
    for i in range(len(x)):
        if x[i] > y[j]:
            z+=[i]
            j+=1
            if j==len(y):
                break
    return z

class spg:

    def __init__(self):
        pass

    def solve(self,x0, f, g, proj, m, eps, maxit, callback=None):
        """TODO."""
        # print "inside spg"
        alpha_min = 1e-3
        alpha_max = 1e3

        f_hist = np.zeros(maxit)

        results = {
            'feval': 0,
            'geval': 0,
            
        }
        #***********
        # print "init my buffer"
        my_buffer = list([])
        #*************
        def linesearch(x_k, f_k, g_k, d_k, k):
            gamma = 1e-4
            sigma_1 = 0.1
            sigma_2 = 0.9

            f_max = np.max(f_hist[max(0, k - m + 1):k + 1])
            delta = np.dot(g_k, d_k)

            x_p = x_k + d_k
            lam = 1

            f_p = f(x_p)
            results['feval'] = results['feval'] + 1
            while f_p > f_max + gamma * lam * delta:
                lam_t = 0.5 * (lam**2) * delta / (f_p - f_k - lam * delta)
                if lam_t >= sigma_1 and lam_t <= sigma_2 * lam:
                    lam = lam_t
                else:
                    lam = lam / 2.0
                x_p = x_k + lam * d_k
                f_p = f(x_p)
                results['feval'] = results['feval'] + 1

            return lam

        # If x_0 \not\in \Omega, replace x_0 by P(x_0)
        x = proj(np.copy(x0))

        f_new = f(x)
        g_new = g(x)
        results['feval'] = results['feval'] + 1
        results['geval'] = results['geval'] + 1
        d = proj(x - g_new) - x
        if np.max(d) <= 0:
            alpha = alpha_max
        else:
            alpha = min(alpha_max, max(alpha_min, 1 / np.max(d)))

        results['bestF'] = None
        results['bestX'] = np.copy(x)

        for k in range(maxit):
            # print(x)
            f_k = f_new
            if False:#True:
	            print('Iter:',k,', x: ', x, ', f(x): ', f(x))
            
            g_k = g_new
            # print(g_new)
            f_hist[k] = f_k
            if results['bestF'] is None or f_k < results['bestF']:
                results['bestF'] = f_k
                results['bestX'][:] = x
                # print('result',f_k)
                # print('mu alpha',x)

            if callback:
                callback(k, results['bestF'])
            d = proj(x - alpha * g_k) - x
            if np.linalg.norm(d) < eps:
                break

            lam = linesearch(x, f_k, g_k, d, k) or 1.0
            s = lam * d
            x += s
            f_new = f(x)
            g_new = g(x)
            #**********
            # print "append"
            my_buffer.append(f_new)
            # print(f_new)

            #**********
            results['feval'] = results['feval'] + 1
            results['geval'] = results['geval'] + 1
            y = g_new - g_k
            beta = np.dot(s, y)
            if beta <= 0:
                alpha = alpha_max
            else:
                alpha = min(alpha_max, max(alpha_min, np.dot(s, s) / beta))
        #**********************
        results['buffer']=my_buffer
        # plt.plot(my_buffer)
        # plt.show()
        # exit()
        #*********************
        # print "exiting spg"
        return results

class Events:

    def __init__(self, d, beta):

        self.d=d
        self.beta = beta


    def Init_data_structures(self, event_time, event_type):
            
        self.n = event_time.shape[0]

        mask = np.zeros((self.n, self.d))
        mask[np.arange(self.n),event_type] = 1 
        mask = mask.T

        final_time = event_time[-1]
        time_decay = 1-np.exp(-self.beta*(final_time - event_time))

        dt = event_time[:,None] - event_time[None,:]
        dt = self.beta*np.exp(-self.beta*dt)

        mask_lower_tri = np.tril(np.ones((self.n, self.n), dtype=int), -1)
        dt = np.multiply(mask_lower_tri, dt)
        
        influence = mask.dot(dt.T)

        return (mask, time_decay, influence, dt)
    

class OptLogLikelihood:

    def __init__(self,event_times, event_types,d=None, beta=None, event_obj=None):
        
        
        self.d=d
        self.beta=beta
        self.n=event_times.shape[0]
        
        self.mask, self.time_decay, self.influence, _ = \
            event_obj.Init_data_structures(event_times, event_types)
        self.influence = np.vstack((\
            np.ones((1,self.n)), self.influence\
        ))

        self.final_time=event_times[-1]
        

    def optimize_X(self):

        X = np.zeros((self.d, self.d+1))

        for i in range(self.d):
            
            x,fval = self.optimize(self.mask[i], \
                self.influence, self.time_decay, self.final_time)
            X[i]=x
        
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

    def CalculateP( self, A):
        
        
        P = np.multiply(\
            self.mask.T.dot( A ).dot(self.mask),\
            self.dt\
            )

        P = np.divide( P , \
            self.lambda_t[:,None]
            )

        return P  

    def CalculateA(self,P):

        divisor = self.mask.dot(self.time_decay)

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
            A = self.CalculateA(P)
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
