# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script contains functions for Module 3 (predicting gene expressoin from chromatin profile) of Seal framework.


## Module
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy import stats
sys.path.insert(0, 'function/')
import early_stop
import exp_data
import weighted_loss


## Pre-train model architecture for predicting expression of general context   
class GeneralContextModel(nn.Module):
    ## 0. Input arguments 
        # input_size (int): number of input features 
        # latent_size (int): numbers of hidden neurons
        # output_size (int): number of general contexts   

    ## 1. Neural network of one hidden layer
    def __init__(self, input_size, latent_size, output_size):
        super(GeneralContextModel, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, output_size)
            )
        
    ## 2. Forward function
    def forward(self, x):
        y = self.layers(x)
        return y


## Fine-tune model architecture for predicting expression of specific context 
class SpecificContextModel(nn.Module):
    ## 0. Input arguments
        # trained_model (pyTorch model): pre-trained model for predicting expression of general context 
        # latent_size (int): numbers of hidden neurons
        # output_size (int): number of specific contexts

    ## 1. Update neural network configuration 
    def __init__(self, trained_model, latent_size, output_size):
        super(SpecificContextModel, self).__init__()
        # freeze input->hidden layer weights  
        for param in trained_model.parameters():
            param.requires_grad = False
        # substitute output layer predicting expression of specific contexts
        new_layers = list(trained_model.layers)[:-1]
        new_layers.append(nn.Linear(latent_size, output_size))
        self.layers = nn.Sequential(*new_layers)

    ## 2. Forward function 
    def forward(self, x):
        y = self.layers(x)
        return y


## This function trains a neural network transfer learning model predicting gene expression from chromatin profile 
def train_tl_model(X_train, y_train, mode, N_latent = None, learned_model = None, model_name = 'checkpoint.pt', learning_rate = 1e-4, l2_lambda = 1e-5, group_file = 'NA', spec_rate = 2, valid_prop = 0.125, N_batch = 32, patience = 20, max_epoch = 200):
    ## 0. Input argument: 
        # X_train (numpy 2D array): array containing feature values of training data 
        # y_train (numpy 1D array): array containing response/label values of combined training data  
        # mode (str): mode of training: 'pre-train' or 'fine-tune'
        # N_latent (int): numbers of hidden neurons 
        # learned_model (pyTorch model): pre-trained model to be fine-tuned (only needed when model is 'fine-tune')
        # model_name (str): output file storing trained model
        # learning_rate (float): learning rate for training
        # l2_lambda (float): L2 regularization factor of loss function
        # group_file (str): name of file containing group info of specific contexts 
        # spec_rate (float): specificity rate for assigning weight to genes (only needed when group_file is NOT 'NA')
        # valid_prop (float): proportion of testing sample*s among combined training and testing data 
        # N_batch (int): number of mini-batches 
        # patience (int): maximal number of epochs to run after testing loss stop decreasing, i.e. optimal model is reached 
        # max_epoch (int): maximal number of epochs to run before stopping if an optimal testing loss cannot be reached 

    ## 1. 
    w_train = weighted_loss.compute_gene_weight(y_train, group_file, spec_rate)

    ## 2. Format training and validation data
    # split data into training and validation according to valid_prop
    X_learn, X_valid, y_learn, y_valid, w_learn, w_valid = train_test_split(X_train, y_train, w_train, test_size = valid_prop, random_state = 0)
    # format feature and label data of training data, then generate data loader accroding to N_batch 
    tl_learn_data = exp_data.exp_dataformat(X_learn, y_learn, w_learn)
    tl_learn_data_loader = torch.utils.data.DataLoader(tl_learn_data, batch_size = N_batch, shuffle = True)
    # format feature and label data of validation data  
    tl_valid_data = exp_data.exp_dataformat(X_valid, y_valid, w_valid)

    ## 3. Construct TL neural network model using the specified hyperparameters
    # define structure of whole neural network  
    if mode == 'pre-train':
        tl_model = GeneralContextModel(X_train.shape[1], N_latent, y_train.shape[1])
    else:
        tl_model = SpecificContextModel(learned_model, N_latent, y_train.shape[1])
    # use Adam optimizer
    tl_optim = torch.optim.Adam(tl_model.parameters(), lr = learning_rate, weight_decay = l2_lambda) 
    # define early stop function
    tl_stop = early_stop.stop(patience = patience, model_name = model_name)
    # 
    tl_valid_loss = nn.MSELoss()

    ## 4. Train TL model
    # perform training by epoch until early stopping criterion is reached  
    epoch = 0
    learn_loss = []
    valid_loss = []
    while tl_stop.early_stop == False:
        epoch += 1
        # set model to training mode 
        tl_model.train()
        # iterate by mini-batch, perform forward and backward propogation, keep track of training/validation loss 
        ep_learn_loss = 0 
        for i, batch_data in enumerate(tl_learn_data_loader, 0):
            # get feature and response data of current batch
            batch_feature, batch_label, batch_weight = batch_data
            # set the gradients to 0
            tl_optim.zero_grad()
            # perform forward propogation to compute predicted output 
            batch_pred = tl_model(batch_feature)
            # compute loss of current batch
            batch_loss = weighted_loss.weighted_mse_loss(batch_pred, batch_label, batch_weight)
            ep_learn_loss += float(batch_loss.data)
            # perform backward propogation
            batch_loss.backward()
            # perform optimization
            tl_optim.step()
        # average computed training loss over all mini-batches, store the average  
        learn_loss.append(ep_learn_loss/(i+1))
        # implement current model to validation data, perform forward propogation to compute predicted output 
        valid_y_pred = tl_model(tl_valid_data.features)
        # compute validation loss
        ep_valid_loss = weighted_loss.weighted_mse_loss(valid_y_pred, tl_valid_data.labels, tl_valid_data.weights)
        valid_loss.append(float(ep_valid_loss.data))
        # check if early stop criterion has been met 
        tl_stop(float(ep_valid_loss.data), tl_model, tl_optim)
        # if so, load the last checkpoint with the best model
        if tl_stop.early_stop:
            stop_point_state = torch.load(model_name)
            tl_model.load_state_dict(stop_point_state['model_state_dict'])
            tl_optim.load_state_dict(stop_point_state['optimizer_state_dict'])
            final_valid_y_pred = tl_model(tl_valid_data.features)
            break
        # stop training if the maximum epoch is reached    
        if epoch == max_epoch:
            final_valid_y_pred = valid_y_pred
            break
    # 
    final_valid_loss1 = tl_valid_loss(final_valid_y_pred, tl_valid_data.labels)
    final_valid_loss = float(final_valid_loss1.data)
    # store training and validation loss of every epoch in data frame form
    train_epoch = np.arange(1, epoch+1)
    train_summary = pd.DataFrame({'epoch': train_epoch, 'training_loss': learn_loss,  'validation_loss': valid_loss})

    return tl_model, train_summary, final_valid_loss


## This function evaluates the performance (mean squared error and spearman correlation) of trained transfer learning model on testing dataset
def evaluate_tl_model(tl_model, X_eval, y_eval):
    ## 0. Input arguments
        # tl_model (pyTorch model): trained neural network model to be evaluated 
        # X_eval (numpy 2D array): array containing feature values of testing data 
        # y_eval (numpy 1D array): array containing response/label values of testing data  

    ## 1. Implement trained model on testing data to generate predicted output
    tl_eval_data = exp_data.exp_dataformat(X_eval, y_eval, torch.zeros(y_eval.shape))
    eval_feat, eval_res, _ = tl_eval_data.features, tl_eval_data.labels, tl_eval_data.weights
    tl_model.eval()
    eval_y_pred = tl_model(eval_feat)
    
    ## 2. Evaluate performance of trained model
    Ne = eval_y_pred.shape[1]
    eval_df = pd.DataFrame(index = np.arange(Ne), columns = ['mse_loss', 'spearman_r'])
    tl_loss = nn.MSELoss()
    for re in range(0, Ne):
        # compute MSELoss between predicted tissue expression and observed tissue expression
        eval_df.iloc[re, 0] = float(tl_loss(eval_y_pred[:, re], eval_res[:, re]).data)
        # compute spearman correlation between predicted tissue expression and observed tissue expression 
        re_pred = np.array(eval_y_pred[:, re].data).flatten()
        re_res = np.array(eval_res[:, re].data).flatten()
        eval_df.iloc[re, 1], _ = stats.spearmanr(re_pred, re_res)
   
    return eval_df


## This function summarizes the performance of Seal transfer learning model (pre-train and fine-tune) 
def summarize_tl_model(out_fd, n_latent, pt_perf_df, ft_perf_df):
    ## 0. Input arguments 
        # out_fd (str): folder storing the model output files  
        # n_latent (int): numbers of hidden neurons
        # pt_perf_df (pandas DataFrame): data frame containing the testing performance metrics of pre-train model 
        # ft_perf_df (pandas DataFrame): data frame containing the testing performance metrics of fine-tune model 

    ## 1. Obtain file names for pre-train and fine-tune model
    out_pt_file = out_fd + '_pt_model.pt' 
    out_ft_file = out_fd + '_ft_model.pt'

    ## 2. Compute the median performance metrics across all contexts 
    pt_med_perf = pt_perf_df.median(axis = 0).values
    n_pt_tissues = pt_perf_df.shape[0]
    pt_tissues = ','.join(pt_perf_df.index)
    ft_med_perf = ft_perf_df.median(axis = 0).values
    n_ft_tissues = ft_perf_df.shape[0]
    ft_tissues = ','.join(ft_perf_df.index)

    ## 3. Construct string list detailing performance of pre-train and fine-tune model
    perf_summary = []
    perf_summary.append('Optimal pretrain model: ' + out_pt_file)
    perf_summary.append('Number of latent features: ' + str(n_latent))
    perf_summary.append('Number of general contexts: ' + str(n_pt_tissues))
    perf_summary.append('Name of general contexts: ' + pt_tissues)
    perf_summary.append('Optimal testing MSE of pretrain model (median of all contexts): ' + str(pt_med_perf[0]))
    perf_summary.append('Optimal testing Spearman r of pretrain model (median of all contexts): ' + str(pt_med_perf[1]))
    perf_summary.append('Optimal finetune model: ' + out_ft_file)
    perf_summary.append('Number of specific contexts: ' + str(n_ft_tissues))
    perf_summary.append('Name of specific contexts: ' + ft_tissues)
    perf_summary.append('Optimal testing MSE of finetune model (median of all contexts): ' + str(ft_med_perf[0]))
    perf_summary.append('Optimal testing Spearman r of finetune model: (median of all contexts): ' + str(ft_med_perf[1]))

    return perf_summary


## This function predicts gene expression based on input chromatin profile  
def predict_expression(tl_model, X_predict):
    ## 0. Input arguments:
        # tl_model (pyTorch model): trained neural network model to be evaluated 
        # X_predict (numpy 2D array): array containing feature values of testing data 

    feat_data = torch.tensor(X_predict, dtype = torch.float)
    tl_model.eval()
    y_pred = tl_model(feat_data)
    y_pred1 = np.array(y_pred.data)

    return y_pred1


## This function load pre-trained model for fine-tuning  
def load_pretrained_model(X_train, y_train, N_latent, trained_folder, new_folder):
    ## 0. Input arguments:
        # X_train (numpy 2D array): array containing feature values of training data 
        # y_train (numpy 1D array): array containing response/label values of combined training data   
        # N_latent (int): numbers of hidden neurons 
        # trained_folder (str): name of folder storing pre-trained model
        # new_folder (str): name of folder storing the new fine-tuned model

    ## 1. Load pre-trained model 
    trained_model_file = trained_folder + '_pt_model.pt' 
    pt_model = GeneralContextModel(X_train.shape[1], N_latent, y_train.shape[1])
    stop_point_state = torch.load(trained_model_file)
    pt_model.load_state_dict(stop_point_state['model_state_dict'])
    # copy pre-trained model file to destination folder 
    new_model_file = new_folder + '_pt_model.pt'
    os.system('cp ' + trained_model_file + ' ' + new_model_file)

    ## 2. Copy training loss summary file of pre-trained model to destination folder 
    trained_training_file = trained_folder + '_pt_training_loss_summary.tsv'
    new_training_file = new_folder + '_pt_training_loss_summary.tsv'
    os.system('cp ' + trained_training_file + ' ' + new_training_file)

    ## 3. Copy testing set performance file of pre-trained model to destination folder 
    trained_testing_file = trained_folder + '_pt_testing_perf.tsv'
    new_testing_file = new_folder + '_pt_testing_perf.tsv'
    os.system('cp ' + trained_testing_file + ' ' + new_testing_file)
    trained_testing_perf = pd.read_csv(trained_testing_file, sep = '\t', header = 0, index_col = 0)

    ## 4. Obtain validation set MSE Loss of pre-trained model 
    trained_valid_file = trained_folder + '_validation_perf.tsv' 
    tvf_df = pd.read_csv(trained_valid_file, sep = '\t', header = None, index_col = 0)
    tvf_loss = tvf_df.loc['pretrain_model_loss', 1]

    return pt_model, trained_testing_perf, tvf_loss


## This function predicts variant effect on expression under specific contexts using the fine-tuned transfer learning model
def predict_var_expression_effect(ref_feat, alt_feat, pt_model_file, ft_model_file, N_latent, N_pt_res, N_ft_res):
    ## 0. Input arguments
        # ref_feat (numpy array): chromatin profile of variant ref alleles 
        # alt_feat (numpy array): chromatin profile of variant alt alleles
        # pt_model_file (str): name of file storing pre-trained model  
        # ft_model_file (str): name of file storing fine-tuned model  
        # N_latent (int): numbers of hidden neurons 
        # N_pt_res (int): number of general contexts that are predicted by the pre-trained model
        # N_ft_res (int): number of specific contexts that are predicted by the fine-tuned model

    ## 1. Load pre-trained and fine-tuned model
    pt_model = GeneralContextModel(ref_feat.shape[1], N_latent, N_pt_res)
    stop_point_state = torch.load(pt_model_file)
    pt_model.load_state_dict(stop_point_state['model_state_dict']) 
    ft_model = SpecificContextModel(pt_model, N_latent, N_ft_res)
    stop_point_state = torch.load(ft_model_file)
    ft_model.load_state_dict(stop_point_state['model_state_dict'])
    
    ## 2. Compute the variant effect on expression as difference between alt allele prediction and ref allele prediction 
    ref_pred_res = predict_expression(ft_model, ref_feat)
    alt_pred_res = predict_expression(ft_model, alt_feat)
    diff_res = alt_pred_res - ref_pred_res

    return diff_res

