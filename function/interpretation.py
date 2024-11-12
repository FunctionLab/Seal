# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script contains functions for Seal interpretation framework


## Module
import sys
import torch
import captum
import numpy as np
import pandas as pd
sys.path.insert(0, 'src/function/')
import exp_effect


## This function assigns the captum function object based on specified interpretation method 
def assign_interpret_function(interpret_method):
    # interpret_method (chr): interpretation method to be implemented

    ## 1.
    func_dict = {
            'saliency': captum.attr.Saliency, 
            'integratedGradients': captum.attr.IntegratedGradients,
            'deeplift': captum.attr.DeepLift,
            'kernalShap': captum.attr.KernelShap,
            'gradientShap': captum.attr.GradientShap,
            'lime': captum.attr.Lime
            }

    return func_dict[interpret_method]


## This function calls specified method to compute ttribution scores of chromatin features for the predicted variant effect 
def compute_attribution_score(ref_feat, alt_feat, pt_model_file, ft_model_file, n_gen, n_spe, n_latent, n_epi_feat, method, target_id):
    ## 0. Input arguments
        # ref_feat (numpy array): chromatin profile of variant ref alleles 
        # alt_feat (numpy array): chromatin profile of variant alt alleles
        # pt_model_file (str): name of file storing pre-trained model  
        # ft_model_file (str): name of file storing fine-tuned model 
        # n_gen (int): number of general contexts that are predicted by the pre-trained model
        # n_spe (int): number of specific contexts that are predicted by the fine-tuned model
        # n_latent (int): numbers of hidden neurons 
        # n_epi_feat (int): number of chromatin features
        # method (str): interpretation method to be implemented ('saliency', 'integratedGradients', 'deeplift', 'kernalShap', 'gradientShap', 'lime')
        # target_id (int): column index of outcome to be interpreted          

    ## 1. Load pre-trained and fine-tuned model
    pt_model = exp_effect.GeneralContextModel(ref_feat.shape[1], n_latent, n_gen)
    stop_point_state = torch.load(pt_model_file)
    pt_model.load_state_dict(stop_point_state['model_state_dict'])
    ft_model = exp_effect.SpecificContextModel(pt_model, n_latent, n_spe)
    stop_point_state = torch.load(ft_model_file)
    ft_model.load_state_dict(stop_point_state['model_state_dict'])

    ## 2.
    ref_feat1 = torch.tensor(ref_feat, dtype = torch.float, requires_grad = True)
    alt_feat1 = torch.tensor(alt_feat, dtype = torch.float, requires_grad = True)
    n_var = ref_feat.shape[0]
    #
    attr_func = assign_interpret_function(method)
    attr_ft = attr_func(ft_model)
    attr_score = attr_ft.attribute(inputs = alt_feat1, baselines = ref_feat1, target = target_id)
    epi_attr_score = np.array(attr_score.reshape(n_var, 10, n_epi_feat).sum(axis = 1).data)

    return epi_attr_score

