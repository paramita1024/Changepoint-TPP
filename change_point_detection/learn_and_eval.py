import time
import torch
from argparse import Namespace
from common import Utils
from models.change_point_methods import differentiable_change_point_detector, Score_statistics, GLR_Hawkes, stochastic_greedy_cpd, Greedy_selection
from itertools import permutations

MethodDict={
    "Greedy_selection":Greedy_selection,
    "stochastic_greedy_cpd":stochastic_greedy_cpd,
    "differentiable_change_point_detector" : differentiable_change_point_detector,
    "Score_statistics" : Score_statistics,
    "GLR_Hawkes" : GLR_Hawkes,
    "None":None
}

class learn_change_point_detection(object):

    def __init__(self, av, config):
        self.av = Namespace()
        Utils.MakeCopy(av,self.av)     
        if av.partition_method=='static':
            partition_str = "Partition_"+av.partitions+"/"
        else:
            partition_str = ""
        sequence_str = "Sequence_"+str(av.seq_no)+"/"
        
        self.av.save_init_dir = config['save_path_for_intial_CPD_model'].replace('dataset',av.data).replace('algorithm',av.method_short) + str(self.av.seed*7) + "/" + partition_str + sequence_str
        self.av.save_intermediate_dir = config['save_path_for_intermediate_CPD_models'].replace('dataset',av.data).replace('algorithm',av.method_short) + str(self.av.seed*7) + "/" + partition_str + sequence_str
        self.av.save_final_dir = config['save_path_for_trained_CPD_model'].replace('dataset',av.data).replace('algorithm',av.method_short) + str(self.av.seed*7) + "/" + partition_str + sequence_str
        self.av.load_CPD_model = config['load_CPD_model_path'].replace('dataset',av.data).replace('algorithm',av.method_short)+ str(self.av.seed*7) + "/" + sequence_str
        self.av.results_dir = self.av.results_path.replace('dataset',av.data).replace('algorithm',av.method_short) + str(self.av.seed*7) + "/" + partition_str + sequence_str 
        self.av.log_file = self.av.results_dir + "log.txt"

        Utils.create_dir(self.av.save_init_dir)
        Utils.create_dir(self.av.save_intermediate_dir)
        Utils.create_dir(self.av.save_final_dir)
        Utils.create_dir(self.av.results_dir)

    def train(self):
        start_time = time.time()
        print("Device used for training: ",self.av.device)
        with open(self.av.log_file, 'a') as f:
            f.write('Device used for training: {}\n'.format(self.av.device))
        dataset = self.av.dataset
        if self.av.load_init_CPD_model == True:
            change_point_detector = torch.load(self.av.load_CPD_model)
            change_point_detector.to(self.av.device)
        elif self.av.load_pre_trained_CPD_model == True:
            change_point_detector = MethodDict[self.av.method]( self.av, dataset[6], dataset[7], dataset[8]).to(self.av.device)
            change_point_detector_dict = change_point_detector.state_dict()
            pre_trained_CPD_dict = torch.load(self.av.load_CPD_model + 'model_parameters'+'.pth', map_location=torch.device('cpu'))
            pre_trained_CPD_dict = {k:v for k,v in pre_trained_CPD_dict.items() if k in change_point_detector_dict}
            change_point_detector_dict.update(pre_trained_CPD_dict)
            change_point_detector.load_state_dict(change_point_detector_dict)
        else:
            change_point_detector = MethodDict[self.av.method]( self.av, dataset[6], dataset[7], dataset[8]).to(self.av.device)
        # exit()

        Utils.save_CPD_model(change_point_detector, self.av, self.av.save_init_dir, init=True)
        change_points = change_point_detector.train( dataset = dataset, results_dir = self.av.results_dir)
        end_time = time.time()
        Utils.save(self.av.results_dir+"Time", end_time - start_time)
        Utils.save_CPD_model_parameters(change_point_detector, self.av.save_final_dir)
        return change_points, change_point_detector, self.av

class evaluate_trained_CPD_model(object):
    def __init__(self, av, config):
        self.av = Namespace()
        Utils.MakeCopy(av,self.av)
        if av.partition_method=='static':
            partition_str = "Partition_"+av.partitions+"/"
        else:
            partition_str = ""
        sequence_str = "Sequence_"+str(av.seq_no)+"/"
        
        self.av.load_CPD_model = config['load_CPD_model_path'].replace('dataset',av.data).replace('algorithm',av.method_short)+ str(self.av.seed*7) + "/" + sequence_str
        self.av.results_dir = self.av.results_path.replace('dataset',av.data).replace('algorithm',av.method_short) + str(self.av.seed*7) + "/"+partition_str + sequence_str
        self.av.log_file = self.av.results_dir + "log.txt"
    
    def compute_mean_detection_error(self, change_points):
        cp_true, cp_estimates = change_points
        error = float('inf')
        comb = None
        if len(cp_true ) == len(cp_estimates):
            list1_permutations = permutations(cp_true, len(cp_estimates))
            for each_permutation in list1_permutations:
                zipped = zip(each_permutation, cp_estimates)
                pre_comb = list(zipped)
                sum_s=0
                for a,b in pre_comb:
                    sum_s+=abs(a-b)
                if(error > sum_s):
                    error = sum_s
                    comb = pre_comb
            error = error/len(cp_estimates)
        else:
            comb = [ cp_true, cp_estimates ]
        print('[Info] The computed error: {}, The combination: {}\n'.format(error,comb))
        with open(self.av.log_file, 'a') as f:
            f.write('[Info] The computed error: {}, The combination: {}\n'.format(error,comb))
        if not isinstance(error,float):
            error = error.item()
        Utils.save(self.av.results_dir+"Error",error)
        # return error
        

    def compute_likelihood_and_CPD_objective(self, change_point_detector, opt):
        device = opt.device
        data_time = opt.dataset[0].to(device)
        data_type = opt.dataset[1].to(device)
        data_feat = opt.dataset[5].to(device)
        num_changepoints = opt.num_changepoints
            
        # fix solution for duff cp
        if method in "differentiable_change_point_detector": 
            time_loss, type_loss, CPD_objective, _ = change_point_detector.forward(data_time, data_type, data_feat, num_changepoints + 1, device)
        if method in "stochastic_greedy_cpd":
            time_loss, type_loss, CPD_objective, _ = change_point_detector.forward(data_time, data_type, data_feat, num_changepoints + 1, device, solution = change_point_detector.solution)
        loss = time_loss + type_loss
        print('[Info] The computed likelihood: {}, The change-point objective: {}\n'.format(loss.item()*-1, CPD_objective))
        with open(self.av.log_file, 'a') as f:
            f.write('[Info] The computed likelihood: {}, The change-point objective: {}\n'.format(loss.item()*-1,CPD_objective))
        return loss.item()*-1, CPD_objective
        

    def load_and_eval_pre_trained_CPD_model(self, opt):
        if self.av.load_pre_trained_CPD_model == True:
            change_point_detector = MethodDict[self.av.method]( self.av, self.av.dataset[6], self.av.dataset[7], self.av.dataset[8]).to(self.av.device)
            change_point_detector_dict = change_point_detector.state_dict()
            pre_trained_CPD_dict = torch.load(self.av.load_CPD_model + 'model_parameters'+'.pth', map_location=torch.device('cpu'))
            pre_trained_CPD_dict = {k:v for k,v in pre_trained_CPD_dict.items() if k in change_point_detector_dict}
            change_point_detector_dict.update(pre_trained_CPD_dict)
            change_point_detector.load_state_dict(change_point_detector_dict)
            device = opt.device
            data_time = opt.dataset[0].to(device)
            data_type = opt.dataset[1].to(device)
            data_feat = opt.dataset[5].to(device)
            num_changepoints = opt.num_changepoints
            time_loss, mark_loss, solution, V, CPD_objective, predictions, (time_nll, type_nll) = change_point_detector.forward(data_time, data_type, data_feat, num_changepoints + 1, device)
            print('[Info] The computed likelihood: {}, The change-point objective: {}\n'.format((time_loss + mark_loss).item()*-1, CPD_objective))
            with open(self.av.log_file, 'a') as f:
                f.write('[Info] The computed likelihood: {}, The change-point objective: {}\n'.format((time_loss + mark_loss)*-1,CPD_objective))
            return (time_loss + mark_loss)*-1, CPD_objective
