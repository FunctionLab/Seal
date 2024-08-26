# !/usr/bin/env python
## created by Yun Hao @FunctionLab 2023
## This script implement the Seal framework for context-specific prediction of variant effect on expression 


## Module
import sys
import argparse
import torch
import numpy as np
import pandas as pd
sys.path.insert(0, 'function/')
import var_data
import epi_effect
import exp_effect 
torch.set_num_threads(20)


## 1. Inputs for implementing Seal model predicting variant effect on expression 
parser = argparse.ArgumentParser(description = 'Process some integers.')
# inputs for module 1&2 
parser.add_argument('--vcf_file', action = 'store', dest = 'vcf_file', type = str)
parser.add_argument('--gene_bed_file', action = 'store', dest = 'gene_bed_file', type = str, default = 'resource/geneanno.pc.sorted.bed')
parser.add_argument('--ref_genome_file', action = 'store', dest = 'ref_genome_file', type = str, default = 'resource/hg19.fa')
parser.add_argument('--seq_model_file', action = 'store', dest = 'seq_model_file', type = str, default = 'resource/deepsea.beluga.pth')
parser.add_argument('--seq_len', action = 'store', dest = 'seq_len', type = int, default = 2000)
parser.add_argument('--n_feature', action = 'store', dest = 'n_feature', type = int, default = 2002)
parser.add_argument('--max_shift', action = 'store', dest = 'max_shift', type = int, default = 800)
parser.add_argument('--batch_size', action = 'store', dest = 'batch_size', type = int, default = 128)
# inputs for module 3
parser.add_argument('--model_info_file', action = 'store', dest = 'model_info_file', type = str)
parser.add_argument('--out_file', action = 'store', dest = 'out_file', type = str)
args = parser.parse_args()

## 1. Read in and process variant VCF file 
var_info_df, nearest_gene_dict = var_data.process_variant_data(args.vcf_file, args.gene_bed_file)

## 2. Load in Seal model for predicting gene expression of specific context of interest  
mid_f = open(args.model_info_file, 'r')
mid_lines = mid_f.readlines()
model_info_dict = {}
for ml in mid_lines:
    ml_s1 = ml.strip()
    ml_s2 = ml_s1.split(': ')
    model_info_dict[ml_s2[0]] = ml_s2[1]

## 3. Iterate by shift size, predicting variant effect at the {shift size} upstream/downstream of variant loc
shift_vec = np.arange(-args.max_shift, args.max_shift + 1, 200)
shift_diff_pred = np.zeros((len(shift_vec), var_info_df.shape[0], int(model_info_dict['Number of specific contexts'])), np.float32)
for isv, sv in enumerate(shift_vec):
    # implement DeepSEA BELUGA model to predict variant effect on chromatin profile (module 1) 
    print('Calculating expression effect at ' + str(sv) + 'bp to the variant')
    sv_ref_pred, sv_alt_pred, sv_ref_match = epi_effect.predict_variant_chromatin_effect(var_info_df, args.ref_genome_file, sv, args.seq_len, args.seq_model_file, args.n_feature, args.batch_size)
    # transform the predicted chromatin profile of TSS-proximal region by exponential function (module 2)
    sv_ref_exp, sv_alt_exp = epi_effect.transform_chromatin_by_exp(sv_ref_pred, sv_alt_pred, sv, var_info_df, nearest_gene_dict, args.n_feature)
    # predict context-specific variant effect on gene expression from chromatin profile 
    shift_diff_pred[isv, :, :] = exp_effect.predict_var_expression_effect(sv_ref_exp, sv_alt_exp, model_info_dict['Optimal pretrain model'], model_info_dict['Optimal finetune model'], int(model_info_dict['Number of latent features']), int(model_info_dict['Number of general contexts']), int(model_info_dict['Number of specific contexts']))
# sum up the predicted effect of all shift size
diff_pred = np.sum(shift_diff_pred, axis = 0)

## 4. Write predicted effect to output tsv file 
var_info_df['reference_genome_match'] = sv_ref_match
diff_pred_df = pd.DataFrame(diff_pred)
diff_pred_df.columns = model_info_dict['Name of specific contexts'].split(',')
diff_pred_df.index  = var_info_df.index 
output_df = pd.concat([var_info_df, diff_pred_df], axis = 1)
output_df.to_csv(args.out_file, sep = '\t', header = True, index = False, float_format = '%.5f')

