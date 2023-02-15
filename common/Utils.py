import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import numpy.random as np_random
import random
import math
# import numpy.linalg as LA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import sys
import shutil
import os
from models.change_point_methods import baseline_change_point_detector

eps=sys.float_info.epsilon

def map_method_to_short(s, pretrain=False):
    if pretrain:
        return {'differentiable_change_point_detector':'DCPD_Pretrain','stochastic_greedy_cpd':'SGCPD_Pretrain', 'GLR_Hawkes':'GLRH', 'Score_statistics':'SS','Greedy_selection':'GS'}[s]
    else:
        return {'differentiable_change_point_detector':'DCPD','stochastic_greedy_cpd':'SGCPD', 'GLR_Hawkes':'GLRH', 'Score_statistics':'SS','Greedy_selection':'GS'}[s]    
    
    
def plot_seeds(array,likelihood_array,CPD_objective_array,save_dir="results"):
    plt.plot(array,likelihood_array,label="likelihood_array")
    plt.legend()    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"likelihood_array.jpg"))
    plt.close()
    plt.plot(array,CPD_objective_array,label="CPD_objective_array")
    plt.legend()    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"CPD_objective_array.jpg"))
    plt.close()

def save_CPD_model(CPD_model, opt, dir, init=False, intermediate=False, epoch=0, final=False):

    if init==True:
        file_name = "Init_CPD_model_" + str(opt.data) + "_lr_" + str(opt.learning_rate) + "_num_cps_" + \
            str(opt.num_changepoints) + "_n_seq_" + str(opt.num_sequences) + "_device_" + str(opt.device) + "_method_" + str(opt.method)

    if intermediate == True:
        file_name = "Intermediate_CPD_model_" + str(opt.data) +"_epoch_"+ str(epoch) + "_lr_" + str(opt.learning_rate) + "_num_cps_" + \
            str(opt.num_changepoints) + "_n_seq_" + str(opt.num_sequences) + "_device_" + str(opt.device) + "_method_" + str(opt.method)
    
    if final == True:
        file_name = "Final_CPD_model_" + str(opt.data) + "_lr_" + str(opt.learning_rate) + "_num_cps_" + \
            str(opt.num_changepoints) + "_n_seq_" + str(opt.num_sequences) + "_device_" + str(opt.device) + "_method_" + str(opt.method)

    if opt.model == 'transformer':
        file_name = file_name + \
                "_THP_" + "_eventEmb_" + str(opt.dim_of_THP) + "_d_hid_" + str(opt.dim_inner_of_THP) + \
                    "_n_layers_" + str(opt.num_layers_of_THP) + "_n_head_" + str(opt.num_head_of_THP) + \
                        "d_k" + str(opt.dim_k_of_THP) + "_d_v_" + str(opt.dim_v_of_THP)

    if isinstance(CPD_model,baseline_change_point_detector):
        file_name = file_name + \
                "_Baseline_" + opt.method +  "_window_length_" + str(opt.window_length) + \
                    "_gamma_" + str(opt.gamma) + "_min_WL_" + str(opt.num_head_of_THP) 

    torch.save(CPD_model,dir+file_name+'.pth')

def save_CPD_model_parameters(CPD_model, dir):
    torch.save(CPD_model.state_dict(), dir+'model_parameters'+'.pth')

def compute_time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se

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
    diff_lamda = (lamda[1:] + lamda[:-1]).squeeze()/2
    return diff_time*diff_lamda

def log_likelihood_array(lamda, time, mask=None):
    if mask is not None:
        log_term = torch.sum(mask[:,1:,:] * torch.log(lamda[:,1:,:]),dim=-1)
        lin_term =  compute_integral_all(torch.sum(lamda, dim=-1), time)
        return - (log_term-lin_term).squeeze()
    else:
        if len(list(lamda.size())) > 1:
            lamda = lamda.squeeze()
        # z=torch.log(lamda[1:])
        # print('lamda time',lamda[-10:].cpu().detach().numpy().flatten())
        return -(torch.log(lamda[1:]) - compute_integral_all(lamda, time))

def log_likelihood_mark_array(mark,target):
    return F.nll_loss(mark, target, reduction='none')

def log_ratios(data_time, lambdas, device, mask=None, data_type=None, mark = None):
    # modified March 26, 2022
    # modified March 26, 2022    
    event_time = torch.unsqueeze(data_time, dim=0)
    if data_type is not None:
        event_type = torch.unsqueeze(data_type, dim=0)

    diff_time = data_time[1:] - data_time[:-1]
    num_changes = lambdas.shape[0]-1
    N = diff_time.shape[0]
    arr_time, arr_type = torch.zeros(num_changes,N).to(device), torch.zeros(num_changes,N).to(device)
    nll_time, nll_type = torch.zeros(num_changes+1,N).to(device), torch.zeros(num_changes+1,N).to(device)

    for i in range(num_changes+1):
        nll_time[i] = log_likelihood_array(lambdas[i], event_time, mask)
        nll_type[i] = log_likelihood_mark_array(mark[i][:-1], event_type[0][1:] - 1)
        
    # print('Time and type 0')
    # print(nll_time[0][-10:].cpu().detach().numpy().flatten())
    # print(nll_type[0][-10:].cpu().detach().numpy().flatten())

    # print('time and type 1')
    # print(nll_time[1][-10:].cpu().detach().numpy().flatten())
    # print(nll_type[1][-10:].cpu().detach().numpy().flatten())

    for i in range(num_changes):
        arr_time[i] = nll_time[i] - nll_time[i+1]
        # print(nll_time[i])   
        if mark is not None:
            arr_type[i] = nll_type[i] - nll_type[i+1]
    arr = arr_time + arr_type
    # print('TIME')
    # print(arr_time[-10:])
    # print('TYPE')
    # print(arr_type[-10:])
    return arr, arr_time, arr_type, nll_time, nll_type

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
        return data

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

def cp_from_scores(scores_times, scores_values, num_change_points):
    idx = -1*num_change_points
    top_cp_idx = np.argsort(scores_values)[idx:]
    cp_estimates = [scores_times[i] for i in top_cp_idx]
    return cp_estimates

def interarrival_cp_plot(event_times, cp_true, cp_estimates, save_dir, file_name="inter_arrival"):
    
    # print(type(event_times))
    event_times = event_times.cpu().detach().numpy()
    plt.plot(event_times[1:],event_times[1:]-event_times[:-1],label="Inter arrival time")
    plt.scatter(cp_true,np.ones_like(cp_true),label="actual change_points",c='red')
    plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()

def intensity_cp_plot(intensities, intensity_times, cp_true, cp_estimates, \
    save_dir, file_name="intensity",\
    flag_cp_detect=True):
    plt.plot(intensity_times, intensities,label='True intensity')
    plt.scatter(cp_true,np.ones_like(cp_true),label="true change points",c='red')
    if flag_cp_detect:
        plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()


def smooth( x ):

    a = np.ones(10)
    tmp = np.convolve(a, x)
    return tmp[:x.shape[0]]


def cp_plot( cp_true, cp_estimates, save_dir, file_name=None, flag_cp_detect=True):
    plt.scatter(cp_true,np.ones_like(cp_true),label="True change points",c='red')
    if flag_cp_detect:
        plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="Estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()


def mark_prediction_cp_plot(event_true_marks, event_times, cp_true, cp_estimates, save_dir, file_name=None, flag_cp_detect=True):
    plt.plot(event_times, event_true_marks,label='True Marks')
    plt.scatter(cp_true,np.ones_like(cp_true),label="True change points",c='red')
    if flag_cp_detect:
        plt.scatter(cp_estimates,np.ones_like(cp_estimates),label="Estimated change points",c='black')
    plt.legend()    
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{file_name}.jpg"))
    plt.close()


def plot_x_y(x, y, save_dir, file_name="plot_x_y"):
    plt.plot(x,y,label=file_name)
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

def get_gradient(data = None, optimizer=None, model=None, if_time_only = False):

    # ************* Passing data through model ************

    event_time, event_type, event_feat = data

    _, _, prediction = model.forward(event_time, event_type, event_feat)
    
    lambdas , time_prediction, type_prediction = prediction

    num_seq, num_events=event_time.size()



    # *********** Compute loss *********************

    nll_loss = torch.sum(log_likelihood_array(lambdas, event_time))

    type_loss = torch.sum(log_likelihood_mark_array(type_prediction[0,:-1,:], event_type[0,1:] - 1)) 

    loss = nll_loss

    if not if_time_only:

        loss += type_loss

    optimizer.zero_grad()
    
    loss.backward()


    # ********** Gradients *************************

    params = list( model.linear_lambdas.parameters()) 

    if not if_time_only:

        params += list( model.type_predictor.parameters())

    param_grad = None

    for param in params:
        
        x = param.grad.data.detach().flatten()

        if param_grad== None:
            
            param_grad = x

        else:

            param_grad = torch.cat((param_grad,x))

    return param_grad

# def get_gradient(data = None, optimizer=None, model=None):

#     # ************* Passing data through model ************

#     event_time, event_type, event_feat = data

#     _, _, prediction = model.forward(event_time, event_type, event_feat)
    
#     lambdas , time_prediction, type_prediction = prediction

#     num_seq, num_events=event_time.size()



#     # *********** Compute loss *********************

#     nll_loss = torch.sum(log_likelihood_array(lambdas, event_time))

#     optimizer.zero_grad()
    
#     nll_loss.backward()


    
#     # ********** Gradients *************************


#     params = list( model.linear_lambdas.parameters())

#     param_grad = None

#     for param in params:
        
#         x = param.grad.data.detach().flatten()

#         # print('x', x.size())

#         if param_grad== None:
            
#             param_grad = x

#         else:

#             param_grad = torch.cat((param_grad,x))

#         # print('param grad', param_grad.size())
    
#     # exit()
#     return param_grad

def Get_inv_fisher_mat(list_of_sub_data, model, optimizer, reg_ss, device): # new_addition

    approx_hessian = None # torch.Tensor([]).to(device)

    for len_interval, sub_data in list_of_sub_data:
        
        dlt_dA = get_gradient( \
            data=sub_data, optimizer=optimizer, model=model)

        dlt_dA = dlt_dA.reshape(-1,1)


        approx_hessian_curr = torch.Tensor(1.0/len_interval).to(device)*dlt_dA.mm(dlt_dA.T)
        
        if approx_hessian==None:
            approx_hessian = approx_hessian_curr
        else:
            approx_hessian += approx_hessian_curr


    approx_hessian_expected = approx_hessian/len(list_of_sub_data)
    # exit()
    n=approx_hessian.shape[0]

    # print(torch.eye(n).get_device())
    # exit()
    return torch.inverse(approx_hessian_expected+reg_ss*torch.eye(n).to(device))


def Get_Score_Stat(dlt_dA, dlt_w_dA, inv_fisher_mat, w):

    n=dlt_dA.shape[0]
    
    dlt_dA = dlt_dA.reshape(-1,1)
    
    dlt_w_dA = dlt_w_dA.reshape(-1,1)
    
    dlt_dA_diff = dlt_dA - dlt_w_dA
    
    I_0_inv_times_dlt = inv_fisher_mat.mm(dlt_dA_diff)
    
    ss_score = (1.0/w)*dlt_dA_diff.T.mm(I_0_inv_times_dlt )

    ss_score = ss_score.flatten()[0]

    return ss_score

def split_data_GLR(data, start_index, cp_index, end_index):

    event_times, event_types, event_features = data 

    data_before_cp = ( event_times[:,start_index:cp_index], \
        event_types[:,start_index:cp_index], event_features[:,start_index:cp_index,:])

    data_after_cp = ( event_times[:,cp_index:end_index], \
        event_types[:,cp_index:end_index], event_features[:,cp_index:end_index,:])

    return data_before_cp, data_after_cp

def split_data_GDCPD(data, start_index, end_index):

    # exit()
    # start_index=10
    # end_index=100
    event_times, event_types, event_features = data 
    # print(event_times.shape)
    # print(event_types.shape)
    # print(event_features.shape)
    # exit()
    return  ( event_times[start_index:end_index], \
        event_types[start_index:end_index], event_features[start_index:end_index,:])

def scale_data(x):
    # print(type(x))
    # exit()
    # print(x[0], x[-1])
    x_min = torch.min(x)#x.min() 
    # print(x_min)
    # exit()
    x_max = torch.max(x)#x.max()
    x_scaled = 100 * (x-x_min)/(x_max-x_min)
    # print(x_min, x_max)
    # print(x_scaled[0], x_scaled[-1])
    
    # exit()
    return x_scaled

# def scale_back(x, scale_attributes):
#     x_min_prev , x_max_prev = scale_attributes
#     x  = x * ((x_max_prev - x_min_prev)/ 100.0) + x_min_prev
#     return x


def split_data_SS(data, cp_index, end_index):

    event_times, event_types, event_features = data 

    data_before_cp = ( event_times[:,:cp_index], \
        event_types[:,:cp_index], event_features[:,:cp_index,:])

    data_after_cp = ( event_times[:,:end_index], \
        event_types[:,:end_index], event_features[:,:end_index,:])

    return data_before_cp, data_after_cp

def Get_mulitple_fragments_of_data(data, num_fragments):
    
    event_times, event_types, event_features = data 
    
    n, data_fragments = event_times.shape[-1],[]
    print(n)
    for i in range(num_fragments):

        i = np.random.randint(n//4,n)

        len_of_interval = (event_times[:,0] - event_times[:,i]).cpu().detach().numpy().flatten()

        data_fragments += [ (len_of_interval,( event_times[:,:i], \
            event_types[:,:i], event_features[:,:i,:])) ]

    return data_fragments
