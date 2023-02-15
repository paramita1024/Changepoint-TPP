from __future__ import division
from cmath import inf
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from common import Utils
from models.cvxpy.cvxlayers import CvxLayerMulti
from models.transformer.Models import Transformer

ModelDict={
    "transformer" : Transformer,
    "None":None
}

class Feed_Forward(nn.Module):

    def __init__(self, in_dim, out_dim):
    
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.layer2 = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
    
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def initialize(self):
    
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer2.weight)

class change_point_detector(nn.Module):
    
    def __init__(self, opt) :
    
        super().__init__()
        self.opt = opt

    def train(self, dataset, results_dir = None):
   
        # if self.opt.perturb:
        #     print('perturbed')
        nn.Module.train(self)
        if self.opt.freeze_transformer_after_pretrain:
            self.model.trainable=False
        num_epochs = self.opt.epochs
        num_changepoints = self.opt.num_changepoints 
        device = self.opt.device
        self.dataset = dataset
        flag_perturb_cp, flag_no_change_cp, perturb_train_count, flag_no_change_cp_arr, check_convergence, set_of_stable_cp, time_to_stop_perturb = False, 0, 0, [], True,{}, False
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.learning_rate)
        data_time, data_type, data_feat, change_points = self.dataset[0].to(device), self.dataset[1].to(device), self.dataset[5].to(device), self.dataset[4]
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not self.opt.whether_global:
            with open(self.opt.log_file, 'a') as f:
                f.write('seed: {}\n'.format(self.opt.seed*7))
            print('[Info] Number of parameters in CPD model: {}'.format(num_params))
            with open(self.opt.log_file, 'a') as f:
                f.write('[Info] Number of parameters in CPD model: {}\n'.format(num_params))
        n, tot_flag, min_val_tot = self.len, 0, float(inf)
        self.changes, loss_array, ll_time_loss_array, ll_mark_loss_array, epoch_array, CPD_array, time_nll_list, type_nll_list, \
        change_points_detected, tot_flag_array = [], [], [], [], [], [], [], [], set([]), []
        # exit()
        for epoch in range(num_epochs):
            print('Epoch', epoch)
            if self.opt.partition_method=='static':
                if epoch > 30:
                    self.opt.partition_method = 'linear'
            # exit()
            time_loss, mark_loss, (CPD_objective, CPD_objective_binary), (solution, V,  predictions, (time_nll, type_nll), ratio_sums, log_ratio_data) = self.forward(data_time, data_type, data_feat, num_changepoints + 1, device, partition_method=self.opt.partition_method, flag_perturb_cp=flag_perturb_cp, perturb_train_count=perturb_train_count)
            # exit()
            loss = time_loss + mark_loss
            tot_nll = time_nll + type_nll
            # - ------------- CP detection --------------
            data_temp = data_time[1:]
            if not self.opt.pre_train_CPD_model:
                last_change_points_detected = set(change_points_detected)
                change_points_detected = [ data_temp[(solution[i+1] > 0.5).cpu().detach().numpy()][0].item() for i in range(num_changepoints)]
                # exit()
                if self.opt.perturb:
                    if flag_perturb_cp:
                        perturb_train_count +=1
                        if perturb_train_count == 10:
                            flag_no_change_cp = 0 
                            min_val_tot = loss.item()
                            tot_flag = 0 
                            check_convergence=True
                            flag_perturb_cp = False
                
                if last_change_points_detected == set(change_points_detected):
                    if self.opt.perturb:
                        flag_no_change_cp += 1
                        if flag_no_change_cp == 25:
                            new_key = '-'.join(map(str, list(last_change_points_detected)))
                            if new_key not in set_of_stable_cp.keys():
                                set_of_stable_cp[new_key]=1
                            else:
                                set_of_stable_cp[new_key] += 1
                                if set_of_stable_cp[new_key] == 20:
                                    time_to_stop_perturb=True
                            if time_to_stop_perturb:
                                self.opt.perturb=False 
                            else:
                                check_convergence=False
                                flag_perturb_cp=True
                                perturb_train_count=0
                                flag_no_change_cp=0
                            # print('perturbed')
                self.changes.append(change_points_detected)          
            # exit()
            #----------------Optimize----------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #----------------Book-keeping---------------------
            epoch_array.append(epoch+1)
            ll_time_loss_array.append(time_loss.item())
            ll_mark_loss_array.append(mark_loss.item())
            CPD_array.append(CPD_objective.item())
            loss_array.append(loss.item())  
            tot_flag_array.append(tot_flag)
            flag_no_change_cp_arr.append(flag_no_change_cp)
            #----------------Checking convergence--------------
            if min_val_tot == float(inf):
                min_val_tot = loss.item()
            if check_convergence:
                if loss.item() < min_val_tot:
                    min_val_tot = loss.item()
                    tot_flag=0
                    # Utils.save_CPD_model(self, self.opt, self.opt.save_intermediate_dir, intermediate=True, epoch = epoch+1)
                else:
                    tot_flag+=1            
            
            #---------------Logging-----------
            if not self.opt.whether_global:

                if epoch % self.opt.save_interval == 0  or tot_flag == 50:  
                    if self.opt.pre_train_CPD_model == True:
                        _, marks = predictions
                        lambdas_pred = torch.zeros_like(data_time)
                        lambdas_pred[1:] = solution*torch.sum(nn.Softplus()(self.linear_pretrained(V)), dim=-1).squeeze()[1:]
                        lambdas_pred[0] = lambdas_pred[1]
                        marks_pred= torch.zeros_like(data_time)
                        marks_pred[1:] += (solution*torch.argmax(marks[:-1], dim=1, keepdim=False)).squeeze()
                        marks_pred[0] = marks_pred[1]
                        print("Epoch: {}     Loss: {:.2f}".format(epoch, loss.item()))
                        with open(self.opt.log_file, 'a') as f:
                            f.write("Epoch: {} Loss: {:.2f}\n".format(epoch, loss.item()))
                        print('Epoch', epoch, ' ', datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"), ' Loss ', loss.item())
                    else: #--------------- Not pretrain -----------------------
                        _,_,_, (_, _,  predictions, (_, _), _, _) = self.forward(data_time, data_type, data_feat, num_changepoints + 1, device)
                        lambdas, marks = predictions
                        lambdas_pred, marks_pred = torch.zeros_like(data_time), torch.zeros_like(data_time)
                        for i in range(num_changepoints+1):
                            lambdas_pred[1:] += (solution[i] - solution[i+1])*lambdas[i][1:]
                            lambdas_pred[0] = lambdas_pred[1]                    
                            marks_pred[1:] += (solution[i] - solution[i+1])*torch.argmax(marks[i][:-1], dim=1, keepdim=False)
                            marks_pred[0] = marks_pred[1]
                        values_time, values_type, nll_time_arr, nll_type_arr = log_ratio_data
                        log_ratio_data_cpu = values_time.cpu().detach().numpy(), values_type.cpu().detach().numpy(), nll_time_arr.cpu().detach().numpy(), nll_type_arr.cpu().detach().numpy()
                        # ---------- Logging -------------------------------
                        Utils.save(results_dir+"change_points",self.changes)
                        Utils.save(results_dir+"log_ratio_data_"+str(epoch),log_ratio_data_cpu)
                        print("Epoch: {}     Loss: {:.2f}     change-point objective: {:.2f}     change-point: {}".format(epoch, loss.item(), CPD_objective.item(), change_points_detected))
                        print('Epoch', epoch, ' ', datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
                        with open(self.opt.log_file, 'a') as f:
                            f.write("Epoch: {} Loss: {:.2f} change-point objective: {:.2f} change-point: {}\n".format(epoch, loss.item(), CPD_objective.item(), change_points_detected))
                    #---------------- Common plots ----------------------------
                    time_nll_list += [time_nll.cpu().detach().numpy()]
                    type_nll_list += [type_nll.cpu().detach().numpy()]
                    Utils.save(results_dir+"nll",{"time":time_nll_list, "type":type_nll_list}) 
                    Utils.save(results_dir+"loss",{"time":ll_time_loss_array,"type":ll_mark_loss_array})
                    Utils.save(results_dir+'predictions',{'lambda':lambdas_pred, 'marks':marks_pred, 'marks_prob':marks})
                    Utils.save(results_dir+'tot_flag',tot_flag_array)
                    Utils.save(results_dir+'no_change_cp',flag_no_change_cp_arr)
                    
                # Utils.save_CPD_model(self, self.opt, self.opt.save_intermediate_dir, intermediate=True, epoch = epoch+1)

            if check_convergence and tot_flag==50:
                break
            
            # print('Epoch', epoch, ' ', datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"), ' Loss ', loss.item())
        # ---------------- Plotting and logging -----------------------------
        # for pre-training
        if self.opt.whether_global:
            cp_index = [torch.where(solution[i+1] > 0.5)[0][0].item() for i in range(num_changepoints)]
            cp_score = CPD_objective_binary.cpu().detach().numpy()
            # print(change_points_detected)
            # print(cp_index)
            # print(cp_score)
            # exit()
            return change_points, change_points_detected, cp_index, cp_score
        if self.opt.pre_train_CPD_model == True:
            return change_points, change_points
        else:
            return change_points, change_points_detected

class change_point_detector_outer(change_point_detector):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt=opt

    def train(self, dataset, results_dir=None):
        
        if self.opt.whether_global:
            device = self.opt.device
            global_num_changepoints = self.opt.num_changepoints
            self.opt.num_changepoints=1
            # print(global_num_changepoints)
            # exit()
            data_time = dataset[0].to(device)
            data_type = dataset[1].to(device)
            data_feat = dataset[5].to(device)
            intensities = dataset[2]
            intensity_times = dataset[3]
            cp_true = dataset[4]
            finish_time = data_time[-1]
            scores = []
            # data_time = torch.unsqueeze(data_time, dim=0)
            # data_type = torch.unsqueeze(data_type, dim=0)
            # data_feat = torch.unsqueeze(data_feat,dim=0)
            start_index= 0
            end_index = torch.where(data_time > data_time[start_index] + self.opt.window_length)[0][0]
            while True:
                print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
                data_segment = Utils.split_data_GDCPD((data_time, data_type, data_feat), start_index, end_index) 
                data_segment_time,data_segment_type, data_segment_feat = data_segment
                # print(data_segment_time[0], data_segment_time[-1])
                data_segment_time_scaled= Utils.scale_data(data_segment_time)
                # print(data_segment_time_scaled[0], data_segment_time_scaled[-1])
                
                dataset_segment = [data_segment_time_scaled, data_segment_type, None, None, None, data_segment_feat ]
                change_points, cp_time, cp_index_relative, cp_score  = change_point_detector.train(self,dataset_segment, results_dir=results_dir)
                # print('before', cp_index_relative)
                # print(cp_time, cp_index, cp_score)
                # print(cp_index_relative)
                # print(start_index)
                # exit()
                cp_index = np.array([start_index + i for i in cp_index_relative ])
                # print('after', cp_index)
                # print('cp time before', cp_time)

                cp_time = data_time[cp_index]
                # print('after cp time', cp_time)
                # exit()
                scores += [(cp_index[0],0,cp_score, cp_time[0])]
                print("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time, cp_score))
                with open(self.opt.log_file, 'a') as f:
                    f.write("Index: {}, Time: {}, Score: {}. ".format(cp_index, cp_time, cp_score))
                    f.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
                # ********* Going forward in the sequence ***********
                start_index += self.opt.gamma
                if start_index  >= data_time.shape[0]:
                    break
                if data_time[start_index] + self.opt.window_length >= finish_time:
                    break
                end_index = torch.where(data_time > data_time[start_index] + self.opt.window_length)[0][0]
                
            # save the score
            Utils.save(results_dir+"Scores",scores)
            scores_times, scores_values = \
                [x for _,_,_,x in scores],\
                [x for _,_,x,_ in scores]
            
            # cp from score
            # print(self.opt.num_changepoints)
            self.opt.num_changepoints=global_num_changepoints
            # print(self.opt.num_changepoints)
            # exit()
            cp_estimates = Utils.cp_from_scores(scores_times, scores_values, self.opt.num_changepoints)
            Utils.save(results_dir+"Changepoints",cp_estimates)
            
            # Plotting and logging
            Utils.save_CPD_model(self, self.opt, self.opt.save_final_dir, final=True)
            return cp_true, cp_estimates
        else:
            # print('yes')
            # exit()
            change_points, change_points_detected = change_point_detector.train(self,dataset, results_dir=results_dir)
            # exit()
            return change_points, change_points_detected

class change_point_detector_sg(change_point_detector):
    
    def __init__(self, opt) :
        super().__init__(opt)
        self.opt = opt

    def train(self, dataset, results_dir = None , number_of_changepoint=None):
        n, num_changepoints, pre_trained_CPD_dict = self.len, self.opt.num_changepoints, None
        device = self.opt.device
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.learning_rate)
        
        self.dataset = dataset
        data_time, data_type, data_feat = self.dataset[0].to(device), self.dataset[1].to(device), self.dataset[5].to(device)    
        intensities, intensity_times, change_points = self.dataset[2], self.dataset[3], self.dataset[4]        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        training_data = (data_time, data_type, data_feat, optimizer, None, None)
        if self.opt.pre_train_CPD_model:
            _, pre_train_loss = self.train_with_cp(None, training_data) 
            pre_trained_CPD_dict=self.state_dict()
            pre_trained_CPD_dict = {k:v for k,v in pre_trained_CPD_dict.items() if k in self.state_dict()}
            training_data = (data_time, data_type, data_feat, optimizer, pre_trained_CPD_dict, results_dir+"Pre-Train-Model_")
            Utils.save(results_dir+"Pre-Train-Loss",pre_train_loss)
            Utils.save_CPD_model_parameters(self, results_dir+"Pre-Train-Model_")
            
        start_index = torch.where(data_time > self.opt.window_length)[0][0]
        set_of_candidate_cps = [x + np.rand.randint(0,4) for x in list(range(start_index, n, self.opt.gamma))]
        num_cp_selected, set_of_cp_selected, final_list_of_loss_array, final_list_of_cp_scores, final_list_of_predictions = 0, [], [], [], []
        while num_cp_selected < num_changepoints:
            cp_selected_list, list_of_loss_array, list_of_final_predictions, cp_scores = self.select_cp(set_of_cp_selected, set_of_candidate_cps, training_data, num_changepoints )
            for cp_selected in cp_selected_list:
                set_of_cp_selected += [cp_selected]
                set_of_candidate_cps.remove(cp_selected)
                num_cp_selected += 1
            final_list_of_loss_array += [list_of_loss_array]
            final_list_of_cp_scores += [cp_scores]
            final_list_of_predictions += [list_of_final_predictions]
        change_points_detected = data_time[np.array(set_of_cp_selected)].cpu().detach().numpy().flatten() 
        candidate_cp_times = data_time[np.array(set_of_candidate_cps)].cpu().detach().numpy().flatten()
        Utils.save(results_dir+"Changepoints",{'Changepoints': change_points_detected, 'Changepoint_scores': final_list_of_cp_scores, 'candidate_cp_times': candidate_cp_times})
        Utils.save(results_dir+"Loss",final_list_of_loss_array)
        Utils.save(results_dir+"Predictions", final_list_of_predictions)
        _,_ = self.train_with_cp(set_of_cp_selected, training_data)
        Utils.save_CPD_model(self, self.opt, self.opt.save_final_dir, final=True)
        return change_points, change_points_detected

    def select_cp(self, set_of_cp_selected, set_of_candidate_cps, training_data, num_changepoints ): 
        cp_scores, list_of_loss_array, list_of_final_predictions, data_time = [], [], [], training_data[0]
        for i, candidate_cp in enumerate(set_of_candidate_cps):
            print('*** Candidate CP ',str(candidate_cp),'***', data_time[candidate_cp].item())
            set_of_cp_curr = set_of_cp_selected + [candidate_cp]
            cp_score, state_info = self.train_with_cp(set_of_cp_curr, training_data)
            loss_array, final_prediction = state_info
            cp_scores += [cp_score]
            list_of_loss_array += [loss_array]
            list_of_final_predictions += [final_prediction]
        if self.opt.batch_select:
            curr_cp_list = list(np.argpartition(np.array(cp_scores), -num_changepoints )[-num_changepoints :])
            ######################################### Above line is corrected ################################################
        else:
            cp_index = np.argmax(np.array(cp_scores))
            curr_cp_list = [set_of_candidate_cps[cp_index]]
        return curr_cp_list, list_of_loss_array, list_of_final_predictions, cp_scores

    def train_with_cp(self, set_of_cp_curr, training_data, save_flag=False, save_suffix=False):
        data_time, data_type, data_feat, optimizer, pre_trained_CPD_dict, pretrain_load_path = training_data 
        print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
        tot_flag, min_val_tot, pretrain_now, loss_array, time_loss_array, type_loss_array, grad_array = 0, float(inf), False, [], [], [], []
        if self.opt.pre_train_CPD_model:
            if set_of_cp_curr is None: # pretrain
                num_changepoints, n =  0, self.len
                solution = torch.ones(1,n-1).to(self.opt.device)
                pretrain_now = True 
            else:
                num_changepoints, n =  len(set_of_cp_curr), self.len
                solution = torch.zeros(len(set_of_cp_curr),n-1).to(self.opt.device)
                set_of_cp_curr.sort()
                for i, cp in enumerate(set_of_cp_curr):
                    solution[i, cp:] = 1
                self.solution=solution
                own_state_dict = self.state_dict()
                if self.opt.load_pretrain:
                    pre_trained_CPD_dict=torch.load(pretrain_load_path+'model_parameters.pth')
                    pre_trained_CPD_dict = {k:v for k,v in pre_trained_CPD_dict.items() if k in own_state_dict}    
                own_state_dict.update(pre_trained_CPD_dict)
                self.load_state_dict(own_state_dict)
                if self.opt.freeze_transformer_after_pretrain:
                    self.model.trainable=False
                self.initialize() 
        else:
            if self.opt.random_init:
                self.initialize()    
            num_changepoints, n =  len(set_of_cp_curr), self.len
            solution = torch.zeros(len(set_of_cp_curr),n-1).to(self.opt.device)
            set_of_cp_curr.sort()
            for i, cp in enumerate(set_of_cp_curr):
                solution[i, cp:] = 1
            self.solution=solution
        # nn.Module.train(self)  
        # self.model.train()
        for epoch in range(self.opt.epochs):
            time_loss, type_loss, CPD_objective, predictions = self.forward(data_time, data_type, data_feat, num_changepoints + 1, self.opt.device, solution=solution, pretrain_now=pretrain_now) 
            loss = time_loss + type_loss            
            optimizer.zero_grad()
            loss.backward()
            # params = list( self.linear1.parameters()) + list(self.linear2.parameters())+list(self.linear_mark1.parameters())+list(self.linear_mark2.parameters())
            # param_grad = None
            # for param in params:
            #     x = param.grad.data.detach().flatten()
            #     if param_grad== None:
            #         param_grad = x
            #     else:
            #         param_grad = torch.cat((param_grad,x))
            # grad_array += [torch.norm(param_grad).item()]
            optimizer.step()
            loss_array += [loss.item()]
            time_loss_array += [time_loss.item()]
            type_loss_array += [type_loss.item()]
            if min_val_tot == float(inf):
                min_val_tot = loss.item()
            if loss.item() < min_val_tot:
                min_val_tot = loss.item()
                tot_flag=0
            else:
                tot_flag+=1
            if tot_flag==50:
                break
        print([loss_array[i] for i in range(0,len(loss_array), 50)])
        # print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
        print('CPD', CPD_objective.item(), ', Loss:', loss.item())
        if save_flag:
            Utils.save_CPD_model_parameters(self, results_dir+"Model"+save_suffix)
        
        lambdas_final, mark_final, time_nll_sum, type_nll_sum, log_ratio_data = predictions
        if not pretrain_now:
            values_time, values_type, nll_time_arr, nll_type_arr = log_ratio_data
            log_ratio_data_cpu = values_time.cpu().detach().numpy(), values_type.cpu().detach().numpy(), nll_time_arr.cpu().detach().numpy(), nll_type_arr.cpu().detach().numpy()            
        else:
            log_ratio_data_cpu = []
        final_prediction = (lambdas_final.cpu().detach().numpy().flatten(), mark_final.cpu().detach().numpy().flatten(), time_nll_sum.cpu().detach().numpy().flatten(), type_nll_sum.cpu().detach().numpy().flatten(), log_ratio_data_cpu)
        return CPD_objective.item(), ((loss_array, time_loss_array, type_loss_array, grad_array), final_prediction)
    
class stochastic_greedy_cpd(change_point_detector_sg):
    def __init__(self, opt, num_types=1, len_seq=1, len_feat=0):
        super().__init__(opt)
        self.num_change_points = opt.num_changepoints
        self.safe = opt.safe
        self.num_types = num_types
        self.len_feat = len_feat
        self.device = opt.device
        self.model = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        )
        self.solution = None    
        self.linear_pretrained = Feed_Forward(opt.dim_of_THP,  1)
        self.linear_mark_pretrained = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)
        self.linear1 = Feed_Forward(opt.dim_of_THP, 1)
        self.linear2 = Feed_Forward(opt.dim_of_THP, 1)
        if opt.num_changepoints >= 2:
            self.linear3 = Feed_Forward(opt.dim_of_THP, 1)
        if opt.num_changepoints >= 3:
            self.linear4 = Feed_Forward(opt.dim_of_THP, 1)

        self.linear_mark1 = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)
        self.linear_mark2 = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)
        if opt.num_changepoints >= 2:
            self.linear_mark3 = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)
        if opt.num_changepoints >= 3:
            self.linear_mark4 = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)

        self.len = len_seq

    def initialize(self):

        self.linear1.initialize()
        self.linear2.initialize()
        if self.opt.num_changepoints >= 2:
            self.linear3.initialize()
        if self.opt.num_changepoints >= 3:
            self.linear4.initialize()

        self.linear_mark1.initialize()
        self.linear_mark2.initialize()
        if self.opt.num_changepoints >= 2:
            self.linear_mark3.initialize()
        if self.opt.num_changepoints >= 3:
            self.linear_mark4.initialize()

        
    def forward(self, data_time, data_type, data_feat, num_part, device, solution=None, pretrain_now=False):
        n =  self.len
        event_time = torch.unsqueeze(data_time, dim=0)
        event_type = torch.unsqueeze(data_type, dim=0)
        event_feat = torch.unsqueeze(data_feat,dim=0)

        V, non_pad_mask, prediction = self.model(event_time, event_type, event_feat)
        if self.opt.pre_train_CPD_model and pretrain_now:
            lambdas_pretrained = nn.Softplus()(self.linear_pretrained(V)).squeeze()
            mark_pretrained = F.log_softmax(self.linear_mark_pretrained(torch.cat((V, event_feat),dim=-1)), dim=-1).squeeze()
            solution_pretrained = torch.ones(1,n-1).to(device)
            time_loss = solution_pretrained@Utils.log_likelihood_array(lambdas_pretrained, event_time)
            type_loss = solution_pretrained@Utils.log_likelihood_mark_array(mark_pretrained[:-1], event_type[0][1:] - 1)
            time_nll_pretrained = solution_pretrained * Utils.log_likelihood_array(lambdas_pretrained, event_time)
            type_nll_pretrained = solution_pretrained * Utils.log_likelihood_mark_array(mark_pretrained[:-1], event_type[0][1:] - 1)
            CPD_objective = torch.tensor(0)
            return time_loss, type_loss, CPD_objective, (lambdas_pretrained, mark_pretrained, time_nll_pretrained, type_nll_pretrained, []) 
        else:            
            lambdas = torch.zeros(num_part*V.shape[0],n).to(device)
            lambdas[0] = nn.Softplus()(self.linear1(V)).squeeze()
            lambdas[1] = nn.Softplus()(self.linear2(V)).squeeze()
            if num_part >= 3:
                lambdas[2] = nn.Softplus()(self.linear3(V)).squeeze()
            if num_part >= 4:
                lambdas[3] = nn.Softplus()(self.linear4(V)).squeeze()

            mark = torch.zeros(num_part*V.shape[0],n,self.num_types).to(device)
            mark[0] = F.log_softmax(self.linear_mark1(torch.cat((V, event_feat),dim=-1)), dim=-1).squeeze() 
            mark[1] = F.log_softmax(self.linear_mark2(torch.cat((V, event_feat),dim=-1)), dim=-1).squeeze()
            if num_part >= 3:
                mark[2] = F.log_softmax(self.linear_mark3(torch.cat((V, event_feat),dim=-1)), dim=-1).squeeze()
            if num_part >= 4:
                mark[3] = F.log_softmax(self.linear_mark4(torch.cat((V, event_feat),dim=-1)), dim=-1).squeeze()

            values, values_time, values_type, nll_time_arr, nll_type_arr \
                 = Utils.log_ratios(data_time, lambdas, device, mask=None, data_type=data_type, mark=mark)
            log_ratio_data = (values_time, values_type, nll_time_arr, nll_type_arr)
            solution = torch.cat([torch.ones(1,n-1).to(device), solution, torch.zeros(1, n-1).to(device)])
            time_loss, type_loss, CPD_objective = 0, 0, 0
            lambdas_final, mark_final, time_nll_final, type_nll_final = torch.zeros(n).to(device), torch.zeros(n).to(device), torch.zeros(n-1).to(device), torch.zeros(n-1).to(device)
            for i in range(num_part):
                time_nll = (solution[i] - solution[i+1]) * Utils.log_likelihood_array(lambdas[i], event_time)
                type_nll = (solution[i] - solution[i+1]) * Utils.log_likelihood_mark_array(mark[i][:-1], event_type[0][1:] - 1)
                lambdas_final[1:] += (solution[i] - solution[i+1]) * lambdas[i][1:]
                mark_final[1:] += (solution[i] - solution[i+1]) * torch.argmax(mark[i][:-1], dim=1, keepdim=False) # Utils.log_likelihood_mark_array(mark[i][:-1], event_type[0][1:] - 1)
                lambdas_final[0], mark_final[0] = lambdas_final[1], mark_final[1]                
                time_nll_final += time_nll 
                type_nll_final += type_nll
                time_loss += torch.sum( time_nll )
                type_loss += torch.sum( type_nll )
                if i > 0:
                    if self.opt.cpd_with_time:
                        CPD_objective_part = solution[i]@values_time[i-1] 
                    else:
                        CPD_objective_part = solution[i]@values[i-1]
                    if self.opt.cpd_scale:
                        cp_index = torch.where(solution[i]>0)[0][0]
                        len_interval = event_times[-1] - event_times[cp_index]
                        CPD_objective_part = CPD_objective_part/len_interval
                    CPD_objective += CPD_objective_part
            return time_loss, type_loss, CPD_objective, (lambdas_final, mark_final, time_nll_final, type_nll_final, log_ratio_data) 

class baseline_change_point_detector(change_point_detector):
    
    def __init__(self, opt) :
    
        super().__init__(opt)
        self.opt = opt

    
    def train(self, dataset, results_dir = None, number_of_changepoint=None):
    
        self.results_dir = results_dir
    
        # print('yes')
        # exit()
        device = self.opt.device
        event_times = dataset[0].to(device)
        event_type = dataset[1].to(device)
        intensities = dataset[2]
        intensity_times = dataset[3]
        cp_true = dataset[4]
        event_features = dataset[5].to(device)
        # exit()
        scores = self.forward(event_times,event_type,event_features)

        # save the score
        Utils.save(results_dir+"Scores",scores)
        
        scores_times, scores_values = \
            [x for _,_,_,x in scores],\
            [x for _,_,x,_ in scores]
        
        # cp from score
        cp_estimates = Utils.cp_from_scores(scores_times, scores_values, self.opt.num_changepoints)
        
        # save the changepoints
        Utils.save(results_dir+"Changepoints",cp_estimates)
        # print(cp_estimates)
        
        # Plotting and logging
        Utils.save_CPD_model(self, self.opt, self.opt.save_final_dir, final=True)
        # Utils.interarrival_cp_plot(event_times, cp_true, cp_estimates, self.results_dir)
        # Utils.intensity_cp_plot(intensities, intensity_times, cp_true, cp_estimates, self.results_dir)
        # Utils.LLRatio_plot(dataset, scores, self.results_dir)

        return cp_true, cp_estimates

class Global_DCPD(baseline_change_point_detector):

    def __init__(self, opt=None, num_types=1,  len_seq=1, len_feat=0):
        
        super().__init__(opt)
        self.num_types = num_types
        self.opt = opt
        self.len_feat = len_feat
        self.num_change_points = opt.num_changepoints

        self.model = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)

    def forward(self, data_time, data_type, data_feat=None, num=1, device="cuda"):
        scores_list = []
        event_times = data_time.cpu().detach() # .cpu().detach().numpy()
        data_time = torch.unsqueeze(data_time, dim=0)
        data_type = torch.unsqueeze(data_type, dim=0)
        data_feat = torch.unsqueeze(data_feat,dim=0)
        
        len_seq, finish_time = event_times.shape[0], event_times[-1]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        cp_index, cp_time = self.opt.min_window_length_index, event_times[self.opt.min_window_length_index]
        start_index, end_index = 0, torch.where(event_times > cp_time+self.opt.window_length)[0][0]
        while True:

            print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
            data_segment = Utils.split_data_GDCPD((data_time, data_type, data_feat), start_index, end_index)
            data_segment_time,data_segment_type, data_segment_feat = data_segment
            num_seq, num_events = data_segment_time.size()  
            # **************** Train the model using data ( t - w ) **********************
            min_val_tot_before_cp = float(inf)
            tot_flag_before_cp=0
            #---------------------------
            self.model.initialize()
            self.model.train()
            for epoch in range(self.opt.epochs):
                _, _, prediction = self.model(data_segment_time,data_segment_type, data_segment_feat)
                lambdas , time_prediction, type_prediction = prediction
                nll_loss = torch.sum(Utils.log_likelihood_array(lambdas, data_time_before_cp))  
                type_loss = torch.sum(Utils.log_likelihood_mark_array(type_prediction[0,:-1,:], data_segment_type[0,1:] - 1)) 
                loss = nll_loss +  type_loss # time_loss +
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #----------------Checking convergence--------------
                if min_val_tot_before_cp == float(inf):
                    min_val_tot_before_cp = loss.item()
                
                if loss.item() < min_val_tot_before_cp:
                    min_val_tot_before_cp = loss.item()
                    tot_flag_before_cp=0
                else:
                    tot_flag_before_cp+=1
                if tot_flag_before_cp==25:
                    break
                    
               
            # **************   Score computation ********
            score = 0 #  TBD
            scores_list += [(cp_index,0,score.item(), cp_time)]
            print("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), score.item()))
            with open(self.opt.log_file, 'a') as f:
                f.write("Index: {}, Time: {}, Score: {}. ".format(cp_index, cp_time.item(), score.item()))
                f.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            # ********* Going forward in the sequence ***********
            cp_index += self.opt.gamma # int((start_index+end_index)/2)
            start_index = torch.where(event_times > cp_time - self.opt.window_length)[0][0]
            end_index = torch.where(event_times > cp_time + self.opt.window_length)[0][0]
            if end_index > event_times.shape[0]:
                break
            cp_time = event_times[cp_index]
                    
        return scores_list

class Score_statistics(baseline_change_point_detector):

    def __init__(self, opt=None, num_types=1,  len_seq=1, len_feat=0):
        
        super().__init__(opt)
        self.num_types = num_types
        self.opt = opt
        self.len_feat = len_feat
        self.num_change_points = opt.num_changepoints

        self.model = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)

    def forward(self, data_time, data_type, data_feat=None, num=1, device="cuda"):
        scores_list = []
        event_times = data_time.cpu().detach() # .cpu().detach().numpy()
        data_time = torch.unsqueeze(data_time, dim=0)
        data_type = torch.unsqueeze(data_type, dim=0)
        data_feat = torch.unsqueeze(data_feat,dim=0)
        len_seq, finish_time = event_times.shape[0], event_times[-1]
        list_of_sub_data = Utils.Get_mulitple_fragments_of_data((data_time, data_type, data_feat), self.opt.num_fragments_fisher_mat )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        cp_index, cp_time = self.opt.min_window_length_index, event_times[self.opt.min_window_length_index]
        end_index = torch.where(event_times > cp_time+self.opt.window_length)[0][0]
        while True:

            print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))

            data_before_cp, data_after_cp = Utils.split_data_SS((data_time, data_type, data_feat), cp_index, end_index)
            data_time_before_cp,data_type_before_cp, data_feat_before_cp = data_before_cp
            num_seq_before_cp, num_events_before_cp = data_time_before_cp.size()  
            # **************** Train the model using data ( t - w ) **********************
            self.model.initialize()
            self.model.train()
            min_val_tot_before_cp = float(inf)
            tot_flag_before_cp=0
            for epoch in range(self.opt.epochs):
                _, _, prediction = self.model(data_time_before_cp,data_type_before_cp, data_feat_before_cp)
                lambdas , time_prediction, type_prediction = prediction
                nll_loss = torch.sum(Utils.log_likelihood_array(lambdas, data_time_before_cp))  # returns scalar
                # time_loss = Utils.compute_time_loss(time_prediction, data_time_before_cp)
                # scale_time_loss = 100
                # time_loss = time_loss/scale_time_loss
                type_loss = torch.sum(Utils.log_likelihood_mark_array(type_prediction[0,:-1,:], data_type_before_cp[0,1:] - 1)) # will not work for many sequence
                loss = nll_loss +  type_loss # time_loss +
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #----------------Checking convergence--------------
                if min_val_tot_before_cp == float(inf):
                    min_val_tot_before_cp = loss.item()
                
                if loss.item() < min_val_tot_before_cp:
                    min_val_tot_before_cp = loss.item()
                    tot_flag_before_cp=0
                else:
                    tot_flag_before_cp+=1
                if tot_flag_before_cp==25:
                    break
                    
               
            # **************   Score computation ********
            inv_fisher_mat = Utils.Get_inv_fisher_mat(list_of_sub_data, self.model, optimizer, self.opt.reg_ss, self.opt.device)
            dlt_w_dA = Utils.get_gradient(data=data_before_cp, optimizer=optimizer, model=self.model)
            dlt_dA = Utils.get_gradient(data=data_after_cp, optimizer=optimizer, model=self.model)
            score = Utils.Get_Score_Stat(dlt_dA, dlt_w_dA, inv_fisher_mat, w=self.opt.window_length)
            scores_list += [(cp_index,0,score.item(), (cp_time+self.opt.window_length/2).item())]
            print("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), score.item()))
            with open(self.opt.log_file, 'a') as f:
                f.write("Index: {}, Time: {}, Score: {}. ".format(cp_index, cp_time.item(), score.item()))
                f.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            # ********* Going forward in the sequence ***********
            cp_index = cp_index + self.opt.gamma
            if cp_index > event_times.shape[0]:
                break
            cp_time = event_times[cp_index]
            if cp_time + self.opt.window_length > finish_time:
                break
            end_index = torch.where(event_times > cp_time+self.opt.window_length)[0][0]
            length_next_window = len_seq - cp_index
            if length_next_window < self.opt.min_window_length_index:
                break        
        return scores_list
    
class GLR_Hawkes(baseline_change_point_detector):

    def __init__(self, \
        opt=None, num_types=1,  \
        len_seq=1, len_feat=0):
        
        super().__init__(opt)
        self.num_types = num_types
        self.opt = opt
        self.len_feat = len_feat
        self.num_change_points = opt.num_changepoints

        # init models
        self.model_before_cp = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)

        self.model_after_cp = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)
    
    def forward(self, data_time, data_type, data_feat=None, num=1, device="cuda"):
        
        ll_ratio = []

        event_times = data_time.cpu().detach()  # .numpy()
        data_time = torch.unsqueeze(data_time, dim=0)
        data_type = torch.unsqueeze(data_type, dim=0)
        data_feat = torch.unsqueeze(data_feat,dim=0)
        len_seq, finish_time = event_times.shape[0], event_times[-1]

        optimizer_before_cp, optimizer_after_cp = \
            torch.optim.Adam(self.model_before_cp.parameters(), lr=self.opt.learning_rate),\
            torch.optim.Adam(self.model_after_cp.parameters(), lr=self.opt.learning_rate)
        cp_index = torch.where(event_times > self.opt.window_length)[0][0]
        cp_time = event_times[cp_index]
        start_index, end_index = torch.where(event_times > cp_time-self.opt.window_length)[0][0],\
            torch.where(event_times > cp_time+self.opt.window_length)[0][0]

        while True:
            data_before_cp, data_after_cp = \
                Utils.split_data_GLR((data_time, data_type, data_feat), start_index, cp_index, end_index) 
            
            data_time_before_cp,data_type_before_cp, data_feat_before_cp = data_before_cp
            num_seq_before_cp, num_events_before_cp = data_time_before_cp.size()  
            
            data_time_after_cp,data_type_after_cp, data_feat_after_cp = data_after_cp
            num_seq_after_cp, num_events_after_cp = data_time_after_cp.size()  
            
            # train model with "data before cp"
            self.model_before_cp.initialize()
            self.model_before_cp.train()
            min_val_tot_before_cp = float(inf)
            tot_flag_before_cp=0

            for epoch in range(self.opt.epochs):
                _, _, prediction_before_cp = self.model_before_cp(data_time_before_cp,data_type_before_cp, data_feat_before_cp)
                lambdas_before_cp, time_prediction_before_cp, type_prediction_before_cp = prediction_before_cp
                nll_loss_before_cp = torch.sum(Utils.log_likelihood_array(lambdas_before_cp, data_time_before_cp))  # returns scalar
                type_loss_before_cp = torch.sum(Utils.log_likelihood_mark_array(\
                    type_prediction_before_cp[0,:-1,:], data_type_before_cp[0,1:] - 1)) # will not work for many sequence

                loss_before_cp = nll_loss_before_cp + type_loss_before_cp
                optimizer_before_cp.zero_grad()
                loss_before_cp.backward()
                optimizer_before_cp.step()
                #----------------Checking convergence--------------
                if min_val_tot_before_cp == float(inf):
                    min_val_tot_before_cp = loss_before_cp.item()
                
                if loss_before_cp.item() < min_val_tot_before_cp:
                    min_val_tot_before_cp = loss_before_cp.item()
                    tot_flag_before_cp=0
                else:
                    tot_flag_before_cp+=1
                if tot_flag_before_cp==50:
                    break
            # ******************           Likelihood on data before cp on trained model    *********************
            _, _, prediction_before_cp = self.model_before_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
            lambdas_before_cp, time_prediction_before_cp, type_prediction_before_cp = prediction_before_cp
            nll_loss_before_cp = torch.sum(Utils.log_likelihood_array(lambdas_before_cp, data_time_after_cp))
            type_loss_before_cp = torch.sum(Utils.log_likelihood_mark_array(\
                type_prediction_before_cp[0,:-1,:], data_type_after_cp[0,1:] - 1))
            tot_loss_before_cp = nll_loss_before_cp + type_loss_before_cp
            # *******************           Train model on "data after cp"  ********************
            self.model_after_cp.initialize()
            self.model_after_cp.train()
            min_val_tot_after_cp = float(inf)
            tot_flag_after_cp=0
            for epoch in range(self.opt.epochs):
                _, _, prediction_after_cp = self.model_after_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
                lambdas_after_cp, time_prediction_after_cp, type_prediction_after_cp = prediction_after_cp
                nll_loss_after_cp = torch.sum(Utils.log_likelihood_array(lambdas_after_cp, data_time_after_cp))  # returns scalar
                type_loss_after_cp = torch.sum(Utils.log_likelihood_mark_array(\
                    type_prediction_after_cp[0,:-1,:], data_type_after_cp[0,1:] - 1)) # will not work for many sequence
                loss_after_cp = nll_loss_after_cp + type_loss_after_cp
                optimizer_after_cp.zero_grad()
                loss_after_cp.backward()
                optimizer_after_cp.step()
                #----------------Checking convergence--------------
                if min_val_tot_after_cp == float(inf):
                    min_val_tot_after_cp = loss_after_cp.item()
                
                if loss_after_cp.item() < min_val_tot_after_cp:
                    min_val_tot_after_cp = loss_after_cp.item()
                    tot_flag_after_cp=0
                else:
                    tot_flag_after_cp+=1
                if tot_flag_after_cp==50:
                    break

            # ****************** Likelihood on data after cp on trained model *************
            _, _, prediction_after_cp = self.model_after_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
            lambdas_after_cp, time_prediction_after_cp, type_prediction_after_cp = prediction_after_cp
            nll_loss_after_cp = torch.sum(Utils.log_likelihood_array(lambdas_after_cp, data_time_after_cp))
            type_loss_after_cp = torch.sum(Utils.log_likelihood_mark_array(\
                type_prediction_after_cp[0,:-1,:], data_type_after_cp[0,1:] - 1))
            tot_loss_after_cp = nll_loss_after_cp + type_loss_after_cp
            # ************ Compute LLRatio score *****************************************************
            ll_ratio_score = -(tot_loss_after_cp - tot_loss_before_cp)
            ll_ratio += [(cp_index,end_index,ll_ratio_score.cpu().detach().numpy(), cp_time.item())]
            print("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), ll_ratio_score.cpu().detach().numpy() ))
            print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            with open(self.opt.log_file, 'a') as f:
                f.write("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), ll_ratio_score.cpu().detach().numpy() ))
                f.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            
            cp_index = cp_index + self.opt.gamma
            if cp_index > event_times.shape[0]:
                break
            cp_time = event_times[cp_index]
            if cp_time + self.opt.window_length > finish_time:
                break
            start_index, end_index = torch.where(event_times > (cp_time-self.opt.window_length))[0][0], torch.where(event_times > (cp_time + self.opt.window_length))[0][0] 
            if end_index - cp_index < self.opt.min_window_length_index:
                break
            
        return ll_ratio 


class Greedy_selection(baseline_change_point_detector):

    def __init__(self, \
        opt=None, num_types=1,  \
        len_seq=1, len_feat=0):
        
        super().__init__(opt)
        self.num_types = num_types
        self.opt = opt
        self.len_feat = len_feat
        self.num_change_points = opt.num_changepoints

        # init models
        self.model_before_cp = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)

        self.model_after_cp = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        ).to(self.opt.device)
    
    def forward(self, data_time, data_type, data_feat=None, num=1, device="cuda"):
        
        ll_ratio = []

        event_times = data_time.cpu().detach()  # .numpy()
        data_time = torch.unsqueeze(data_time, dim=0)
        data_type = torch.unsqueeze(data_type, dim=0)
        data_feat = torch.unsqueeze(data_feat,dim=0)
        len_seq, finish_time = event_times.shape[0], event_times[-1]

        optimizer_before_cp, optimizer_after_cp = \
            torch.optim.Adam(self.model_before_cp.parameters(), lr=self.opt.learning_rate),\
            torch.optim.Adam(self.model_after_cp.parameters(), lr=self.opt.learning_rate)
        cp_index = torch.where(event_times > self.opt.window_length)[0][0]
        cp_time = event_times[cp_index]
        start_index, end_index = 0, event_times.shape[0]
        # torch.where(event_times > cp_time-self.opt.window_length)[0][0],\
        #     torch.where(event_times > cp_time+self.opt.window_length)[0][0]

        while True:
            
            data_before_cp, data_after_cp = \
                Utils.split_data_GLR((data_time, data_type, data_feat), start_index, cp_index, end_index) 
            
            data_time_before_cp,data_type_before_cp, data_feat_before_cp = data_before_cp
            num_seq_before_cp, num_events_before_cp = data_time_before_cp.size()  
            
            data_time_after_cp,data_type_after_cp, data_feat_after_cp = data_after_cp
            num_seq_after_cp, num_events_after_cp = data_time_after_cp.size()  
            
            # train model with "data before cp"
            self.model_before_cp.initialize()
            self.model_before_cp.train()
            min_val_tot_before_cp = float(inf)
            tot_flag_before_cp=0

            for epoch in range(self.opt.epochs):
                _, _, prediction_before_cp = self.model_before_cp(data_time_before_cp,data_type_before_cp, data_feat_before_cp)
                lambdas_before_cp, time_prediction_before_cp, type_prediction_before_cp = prediction_before_cp
                nll_loss_before_cp = torch.sum(Utils.log_likelihood_array(lambdas_before_cp, data_time_before_cp))  # returns scalar
                type_loss_before_cp = torch.sum(Utils.log_likelihood_mark_array(\
                    type_prediction_before_cp[0,:-1,:], data_type_before_cp[0,1:] - 1)) # will not work for many sequence

                loss_before_cp = nll_loss_before_cp + type_loss_before_cp
                optimizer_before_cp.zero_grad()
                loss_before_cp.backward()
                optimizer_before_cp.step()
                #----------------Checking convergence--------------
                if min_val_tot_before_cp == float(inf):
                    min_val_tot_before_cp = loss_before_cp.item()
                
                if loss_before_cp.item() < min_val_tot_before_cp:
                    min_val_tot_before_cp = loss_before_cp.item()
                    tot_flag_before_cp=0
                else:
                    tot_flag_before_cp+=1
                if tot_flag_before_cp==50:
                    break
            # ******************           Likelihood on data before cp on trained model    *********************
            _, _, prediction_before_cp = self.model_before_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
            lambdas_before_cp, time_prediction_before_cp, type_prediction_before_cp = prediction_before_cp
            nll_loss_before_cp = torch.sum(Utils.log_likelihood_array(lambdas_before_cp, data_time_after_cp))
            type_loss_before_cp = torch.sum(Utils.log_likelihood_mark_array(\
                type_prediction_before_cp[0,:-1,:], data_type_after_cp[0,1:] - 1))
            tot_loss_before_cp = nll_loss_before_cp + type_loss_before_cp
            
            # *******************           Train model on "data after cp"  ********************
            self.model_after_cp.initialize()
            self.model_after_cp.train()
            min_val_tot_after_cp = float(inf)
            tot_flag_after_cp=0
            for epoch in range(self.opt.epochs):
                _, _, prediction_after_cp = self.model_after_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
                lambdas_after_cp, time_prediction_after_cp, type_prediction_after_cp = prediction_after_cp
                nll_loss_after_cp = torch.sum(Utils.log_likelihood_array(lambdas_after_cp, data_time_after_cp))  # returns scalar
                type_loss_after_cp = torch.sum(Utils.log_likelihood_mark_array(\
                    type_prediction_after_cp[0,:-1,:], data_type_after_cp[0,1:] - 1)) # will not work for many sequence
                loss_after_cp = nll_loss_after_cp + type_loss_after_cp
                optimizer_after_cp.zero_grad()
                loss_after_cp.backward()
                optimizer_after_cp.step()
                #----------------Checking convergence--------------
                if min_val_tot_after_cp == float(inf):
                    min_val_tot_after_cp = loss_after_cp.item()
                
                if loss_after_cp.item() < min_val_tot_after_cp:
                    min_val_tot_after_cp = loss_after_cp.item()
                    tot_flag_after_cp=0
                else:
                    tot_flag_after_cp+=1
                if tot_flag_after_cp==50:
                    break

            # ****************** Likelihood on data after cp on trained model *************
            _, _, prediction_after_cp = self.model_after_cp(data_time_after_cp,data_type_after_cp, data_feat_after_cp)
            lambdas_after_cp, time_prediction_after_cp, type_prediction_after_cp = prediction_after_cp
            nll_loss_after_cp = torch.sum(Utils.log_likelihood_array(lambdas_after_cp, data_time_after_cp))
            type_loss_after_cp = torch.sum(Utils.log_likelihood_mark_array(\
                type_prediction_after_cp[0,:-1,:], data_type_after_cp[0,1:] - 1))
            tot_loss_after_cp = nll_loss_after_cp + type_loss_after_cp
            # ************ Compute LLRatio score *****************************************************
            ll_ratio_score = -(tot_loss_after_cp - tot_loss_before_cp)
            ll_ratio += [(cp_index,end_index,ll_ratio_score.cpu().detach().numpy(), cp_time.item())]
            print("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), ll_ratio_score.cpu().detach().numpy() ))
            print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            with open(self.opt.log_file, 'a') as f:
                f.write("Index: {}, Time: {}, Score: {}".format(cp_index, cp_time.item(), ll_ratio_score.cpu().detach().numpy() ))
                f.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")+"\n")
            
            cp_index = cp_index + self.opt.gamma
            if cp_index >= event_times.shape[0]:
                break
            cp_time = event_times[cp_index]
            if cp_time + self.opt.window_length > finish_time:
                break
            # start_index, end_index = torch.where(event_times > (cp_time-self.opt.window_length))[0][0], torch.where(event_times > (cp_time + self.opt.window_length))[0][0] 
            if end_index - cp_index < self.opt.min_window_length_index:
                break
            
        return ll_ratio 

class differentiable_change_point_detector(change_point_detector_outer):
    
    def __init__(self, opt, num_types=1, len_seq=1, len_feat=0):
        super().__init__(opt)
        self.num_change_points = opt.num_changepoints
        self.safe = opt.safe
        self.num_types = num_types
        self.len_feat = len_feat

        self.model = ModelDict[self.opt.model](
            num_types=self.num_types,
            dim_of_THP=opt.dim_of_THP,
            dim_inner_of_THP=opt.dim_inner_of_THP,
            num_layers_of_THP=opt.num_layers_of_THP,
            num_head_of_THP=opt.num_head_of_THP,
            dim_k_of_THP=opt.dim_k_of_THP,
            dim_v_of_THP=opt.dim_v_of_THP,
            dropout=opt.dropout,
            future_of_THP=opt.future_of_THP,
            device = opt.device,
            len_feat = self.len_feat
        )
        
        # for pre-training
        if self.opt.pre_train_CPD_model == True:
            self.linear_pretrained = Feed_Forward(opt.dim_of_THP,  1)
            self.linear_mark_pretrained = Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types)
        # for pre-training ends here
        else:
            self.linear = [Feed_Forward(opt.dim_of_THP, 1) for _ in range(opt.num_changepoints+1)]
            self.linear_mark = [Feed_Forward(opt.dim_of_THP+self.len_feat, self.num_types) for _ in range(opt.num_changepoints+1)]
        self.len = len_seq
        self.last_solution = torch.zeros(opt.num_changepoints,self.len-1)
                

    def forward(self, data_time, data_type, data_feat, num_part, device, partition_method = 'cvxpy', flag_perturb_cp=False, perturb_train_count=0):
        # print('Inside')
        # print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
        # n=event_time.shape[0]
        # print(data_time.shape)
        n=data_time.shape[0]
        event_time = torch.unsqueeze(data_time, dim=0)
        event_time = event_time.type(torch.FloatTensor)
        event_type = torch.unsqueeze(data_type, dim=0)
        event_feat = torch.unsqueeze(data_feat,dim=0)
        # exit()

        V, non_pad_mask, prediction = self.model(event_time, event_type, event_feat)
        if self.opt.pre_train_CPD_model == True:
            
            lambdas_pretrained = nn.Softplus()(self.linear_pretrained(V)).squeeze()
            mark_pretrained = torch.squeeze(F.log_softmax(self.linear_mark_pretrained(torch.cat((V, event_feat),dim=-1)), dim=-1),0)
            solution_pretrained = torch.ones(1,n-1).to(device)
            ll_loss = solution_pretrained@Utils.log_likelihood_array(lambdas_pretrained, event_time)
            time_loss = Utils.compute_time_loss(prediction[1], event_time)
            scale_time_loss = 10
            time_loss = time_loss/scale_time_loss
            type_loss = solution_pretrained@Utils.log_likelihood_mark_array(mark_pretrained[:-1], event_type[0][1:] - 1)
            CPD_objective = torch.tensor(0)
            tot_loss = ll_loss + type_loss + time_loss
            time_nll_pretrained = solution_pretrained * Utils.log_likelihood_array(lambdas_pretrained, event_time)
            type_nll_pretrained = solution_pretrained * Utils.log_likelihood_mark_array(\
                    mark_pretrained[:-1], event_type[0][1:] - 1)
            # exit()
            return ll_loss, type_loss, (CPD_objective,CPD_objective), (solution_pretrained, V,  (lambdas_pretrained, mark_pretrained), (time_nll_pretrained, type_nll_pretrained) , None, None)
        else:            
            lambdas = torch.zeros(num_part*V.shape[0],n).to(device)
            mark = torch.zeros(num_part*V.shape[0],n,self.num_types).to(device)
            for i in range(num_part):
                lambdas[i] = nn.Softplus()(self.linear[i](V)).squeeze()
                mark[i] = torch.squeeze(F.log_softmax(self.linear_mark[i](torch.cat((V, event_feat),dim=-1)), dim=-1),0)
            
            values, values_time, values_type, nll_time_arr, nll_type_arr \
                 = Utils.log_ratios(data_time, lambdas, device, mask=None, data_type=data_type, mark=mark)
            log_ratio_data = (values_time, values_type, nll_time_arr, nll_type_arr)

            ratio_sums=[]
            if partition_method=='static':
                solution = torch.zeros(num_part-1,n-1).to(device)
                if self.opt.num_changepoints==1:
                    list_partitions = [int(self.opt.partitions)]
                else:
                    list_partitions = [int(x) for x in self.opt.partitions.split('-')]
                for i, cp_index  in enumerate(list_partitions):
                    solution[i,cp_index]=1
                    # exit()
            if partition_method == 'linear':  
                if self.opt.cpd_with_time:
                    ratios = values_time[0].cpu().detach().numpy()
                else:
                    ratios = values[0].cpu().detach().numpy()
                # print(ratios[:10])
                # print(ratios[-10:])                
                ratios = ratios[::-1]
                ratio_sums = np.cumsum(ratios)
                ratio_sums = ratio_sums[::-1]
                cp_index = np.argmax(ratio_sums)
                print('cp index', cp_index)
                solution = torch.zeros(1,n-1).to(device)
                solution[:,cp_index:] = 1
            if partition_method=='cvxpy':
                if flag_perturb_cp:
                    if perturb_train_count==0:
                        lower_index, higher_index = int( 0.2 * n ), int( 0.8 * n )
                        cps = set()
                        while len(cps) < num_part - 1:
                            cps.add(np.random.randint(lower_index, higher_index))
                        cps = sorted(list(cps))
                        print('cps', cps)
                        solution = torch.zeros( num_part - 1 ,n-1).to(device)
                        for i,cp_index in zip(range(num_part-1),list(cps)):
                            solution[i,cp_index:] = 1 
                        self.last_solution = torch.clone(solution).detach()
                    else:
                        solution = torch.clone(self.last_solution)
                else:
                    self.cvx_layer = CvxLayerMulti(n-1, num_part-1, self.safe)
                    if self.opt.cpd_with_time:
                        solution = self.cvx_layer.out(values_time)
                    else:
                        solution = self.cvx_layer.out(values)
            solution = torch.cat([torch.ones(1,n-1).to(device), solution, torch.zeros(1, n-1).to(device)])

            time_loss, type_loss, CPD_objective, CPD_objective_biary = 0, 0, 0, 0
            time_nll_sum, type_nll_sum = torch.zeros(n-1).to(device), torch.zeros(n-1).to(device) 
            # print('solution',solution.shape)
            # print('values', values.shape)
            # exit()

            for i in range(num_part):
                time_nll = (solution[i] - solution[i+1]) * Utils.log_likelihood_array(lambdas[i], event_time)
                type_nll = (solution[i] - solution[i+1]) * Utils.log_likelihood_mark_array(mark[i][:-1], event_type[0][1:] - 1)
                time_nll_sum, type_nll_sum = time_nll_sum + time_nll, type_nll_sum + type_nll
                time_loss += torch.sum( time_nll )
                type_loss += torch.sum( type_nll )
                if i != 0:
                    CPD_objective += solution[i]@values[i-1]
                    tmp = values[i-1][solution[i]>0.5]
                    # print(tmp.shape)
                    # print(solution[i].min(), solution[i].max(), solution[i][-10:])
                    CPD_objective_biary += torch.sum(values[i-1][solution[i]>0.5])
                    # exit()
            return time_loss, type_loss, (CPD_objective, CPD_objective_biary), (solution, V,  (lambdas, mark), (time_nll_sum, type_nll_sum), ratio_sums, log_ratio_data)
