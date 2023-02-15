import torch
import torch.nn as nn
import configparser
from common import Utils
import argparse
from argparse import Namespace
from change_point_detection.learn_and_eval import learn_change_point_detection, evaluate_trained_CPD_model
        
def main_call(av, dataset, data, config, config_parameters, i):

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--device",                                 type=str,   default='cpu')#uda:0')
    ap.add_argument("--method",                                 type=str,   default='differentiable_change_point_detector')
    ap.add_argument("--model",                                  type=str,   default='transformer')
    ap.add_argument("--load_synthetic_dataset",                 action='store_true')
    ap.add_argument("--load_init_CPD_model",                    action='store_true')
    ap.add_argument("--stochastic_greedy",                      action='store_true')
    ap.add_argument("--batch_select",                           action='store_true')
    ap.add_argument("--whether_global",                         action='store_true')
    ap.add_argument("--cpd_with_time",                          action='store_true')
    ap.add_argument("--cpd_scale",                              action='store_true')
    ap.add_argument("--perturb",                                action='store_true')
    ap.add_argument("--memorize",                               action='store_true')
    ap.add_argument("--vary_next_dist",                         action='store_true')
    ap.add_argument("--linear_solution",                        action='store_true')
    ap.add_argument("--pre_train_CPD_model",                    action='store_true')
    ap.add_argument("--load_pre_trained_CPD_model",             action='store_true')
    ap.add_argument("--only_dataset_generation",                action='store_true')
    ap.add_argument("--results_path",                           type=str,   default=str(config['USER_MACHINE_REAL']['results_path']))
    ap.add_argument("--dataset",                                type=str,   default="Particle")
    ap.add_argument("--real_data_path",                         type=str,   default=str(config['USER_MACHINE_REAL']['real_data_path']))
    ap.add_argument("--learning_rate",                          type=float, default =float(config['DEFAULT']['learning_rate']))
    ap.add_argument("--epochs",                                 type=int,   default=int(config['DEFAULT']['epochs']))
    ap.add_argument("--num_changepoints",                       type=int,   default =int(config['DEFAULT']['num_changepoints']))
    ap.add_argument("--num_sequences",                          type=int,   default =int(config['DEFAULT']['num_sequences']))
    ap.add_argument("--save_interval",                          type=int,   default = int(config['DEFAULT']['save_interval']))
    ap.add_argument("--partition_method",                       type=str,   default='cvxpy')
    ap.add_argument("--partitions",                             type=str,   default='1500')
    # stochastic greedy parameters
    ap.add_argument("--random_init",                            action='store_true')
    ap.add_argument("--freeze_transformer_after_pretrain",      action='store_true')
    ap.add_argument("--load_pretrain",                          action='store_true')
    ap.add_argument("--save_states_sgcpd",                      action='store_true')
    # cvxpy parameters
    ap.add_argument("--safe",                                   type=int,   default = int(config_parameters['CVXPY']['safe']))
    ap.add_argument("--seed",                                   type=int,   default = int(config['DEFAULT']['seed']))
    # transformer parameters
    ap.add_argument("--dim_of_THP",                             type=int,   default = int(config_parameters['TRANSFORMER']['dim_of_THP']))
    ap.add_argument("--dim_inner_of_THP",                       type=int,   default = int(config_parameters['TRANSFORMER']['dim_inner_of_THP']))
    ap.add_argument("--num_layers_of_THP",                      type=int,   default = int(config_parameters['TRANSFORMER']['num_layers_of_THP']))
    ap.add_argument("--num_head_of_THP",                        type=int,   default = int(config_parameters['TRANSFORMER']['num_head_of_THP']))
    ap.add_argument("--dropout",                                type=float, default = float(config_parameters['TRANSFORMER']['dropout']))
    ap.add_argument("--dim_k_of_THP",                           type=int,   default = int(config_parameters['TRANSFORMER']['dim_k_of_THP']))
    ap.add_argument("--dim_v_of_THP",                           type=int,   default = int(config_parameters['TRANSFORMER']['dim_v_of_THP']))
    ap.add_argument("--future_of_THP",                          type=int,   default = int(config_parameters['TRANSFORMER']['future_of_THP']))
    # baseline parameters
    ap.add_argument("--window_length",                          type=int,   default = int(config_parameters['BASELINES']['window_length']), help ="window length")
    ap.add_argument("--gamma",                                  type=int,   default = int(config_parameters['BASELINES']['gamma']), help="next start index")
    ap.add_argument("--min_window_length_index",                type=int,   default = int(config_parameters['BASELINES']['min_window_length_index']), help = "minimum number of events in a window")
    ap.add_argument("--reg_ss",                                 type=float, default = float(config_parameters['BASELINES']['reg_ss']), help="reg score statistics")
    ap.add_argument("--num_fragments_fisher_mat",               type=int,   default = int(config_parameters['BASELINES']['num_fragments_fisher_mat']), help = "number of fragments in fischer matrix")
    

    av =  ap.parse_args()
    av.dataset = dataset
    av.data = data
    av.method_short = Utils.map_method_to_short(av.method, av.pre_train_CPD_model)
    av.device = torch.device('cpu')
    av.seq_no = i 
    # if torch.cuda.is_available():
    #     av.device = torch.device('cuda:1')
    #     print('available')
    # else:
    #     av.device = torch.device('cpu')
    #     print("Not available")
    Utils.set_seed(av.seed*7)
    # exit()

    learn = learn_change_point_detection(av, config['USER_MACHINE_REAL'])
    # exit()
    change_points, change_point_detector, opt = learn.train()  
    #return #exit()
    eval = evaluate_trained_CPD_model(av, config['USER_MACHINE_REAL'])
    eval.compute_mean_detection_error(change_points)
    # if av.method in ['differentiable_change_point_detector','stochastic_greedy_cpd']:
    #     eval.compute_likelihood_and_CPD_objective(change_point_detector,opt)
    
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('common/config.ini')
    # seed = config['DEFAULT']['seed']
    # Utils.set_seed(int(seed))
    data_path = config['USER_MACHINE_REAL']['real_data_path']
    config_parameters = configparser.ConfigParser()
    config_parameters.read('common/config_parameters.ini')
    for i,dataset in enumerate(Utils.load_dataset(data_path)):
        if True:#i in [1,4,7,10]:#i in range(0,100):#[0]:#1,2,4,7,10]:
            #print("It is executing")
            #exit()
            #import time
            #start = time.time()
            torch.cuda.empty_cache()    
            av = Namespace()
            dataset_name = data_path.split("/")[-1]
            main_call(av, dataset, dataset_name, config, config_parameters, i)
            #end=time.time()
            #print(end-start, ' seconds')
        
        
