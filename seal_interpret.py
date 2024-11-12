# !u/sr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script interprets the Seal framework to compute attribution scores of chromatin features 


## Module
import sys
import argparse
import torch
import captum
import numpy as np
import pandas as pd
sys.path.insert(0, 'function/')
import var_data
import epi_effect
import interpretation
torch.set_num_threads(20)


## 0. Input for implementing interpretation methods for Seal model 
parser = argparse.ArgumentParser(description = 'Process some integers.')
# input for seal model 
parser.add_argument('--ref_genome_file', action = 'store', dest = 'ref_genome_file', type = str, default = 'resource/hg19.fa')
parser.add_argument('--seq_len', action = 'store', dest = 'seq_len', type = int, default = 2000)
parser.add_argument('--seq_model_file', action = 'store', dest = 'seq_model_file', type = str, default = 'resource/deepsea.beluga.pth')
parser.add_argument('--n_feature', action = 'store', dest = 'n_feature', type = int, default = 2002)
parser.add_argument('--max_shift', action = 'store', dest = 'max_shift', type = int, default = 800)
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', type = int, default = 128)
parser.add_argument('--gene_bed_file', action = 'store', dest = 'gene_bed_file', type = str, default = 'resource/geneanno.pc.sorted.bed')
parser.add_argument('--vcf_file', action = 'store', dest = 'vcf_file', type = str)
parser.add_argument('--model_info_file', action = 'store', dest = 'model_info_file', type = str)
# input for interpretation framework 
parser.add_argument('--interpret_method', action = 'store', dest = 'interpret_method', type = str)
parser.add_argument('--outcome_id', action = 'store', dest = 'outcome_id', type = int)
parser.add_argument('--out_file', action = 'store', dest = 'out_file', type = str)
args = parser.parse_args()

## 1. Read in and process variant VCF file 
var_info_df, nearest_gene_dict = var_data.process_variant_data(args.vcf_file, args.gene_bed_file)
N_var = var_info_df.shape[0]

## 2. Load in Seal model for predicting gene expression of specific context of interest  
mid_f = open(args.model_info_file, 'r')
mid_lines = mid_f.readlines()
model_info_dict = {}
for ml in mid_lines:
    ml_s1 = ml.strip()
    ml_s2 = ml_s1.split(': ')
    model_info_dict[ml_s2[0]] = ml_s2[1]
# obtain model stat
N_latent_feat = int(model_info_dict['Number of latent features'])
N_gen = int(model_info_dict['Number of general contexts'])
N_spe = int(model_info_dict['Number of specific contexts'])

## 3. Iterate by shift size, computing attribution scores of chromatin features for the predicted variant effect at the {shift size} upstream/downstream of variant loc
shift_vec = np.arange(-args.max_shift, args.max_shift + 1, 200)
diff_attri = np.zeros((N_var, args.n_feature), np.float32)
for isv, sv in enumerate(shift_vec):
    # implement DeepSEA BELUGA model to predict variant effect on chromatin profile (Seal module 1) 
    sv_ref_pred, sv_alt_pred, nl_ref_match = epi_effect.predict_variant_chromatin_effect(var_info_df, args.ref_genome_file, sv, args.seq_len, args.seq_model_file, args.n_feature, args.batch_size)
    # transform the predicted chromatin profile of TSS-proximal region by exponential function (Seal module 2)
    sv_ref_exp, sv_alt_exp = epi_effect.transform_chromatin_by_exp(sv_ref_pred, sv_alt_pred, sv, var_info_df, nearest_gene_dict, args.n_feature)
    # call specified method to compute ttribution scores of chromatin features for the predicted variant effect 
    isv_diff_attri = interpretation.compute_attribution_score(sv_ref_exp, sv_alt_exp, model_info_dict['Optimal pretrain model'], model_info_dict['Optimal finetune model'], N_gen, N_spe, N_latent_feat, args.n_feature, args.interpret_method, args.outcome_id)
    diff_attri = diff_attri + isv_diff_attri
    
## 4. Write computed attritbuion score to output npy file 
np.save(args.out_file + '_' + args.interpret_method + '_attribution_score_outcome_' + str(args.outcome_id) + '.npy', diff_attri)

