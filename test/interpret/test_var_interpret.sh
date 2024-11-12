# !/usr/bin/env bash
## created by Yun Hao @FunctionLab 2023
## This script contains command line examples that implement the Seal interpretation framework that computes attribution scores of chromatin features for the predicted variant effect 

# Usage:
# python seal_interpret.py --vcf_file <variant vcf file> --model_info_file <Seal model summary file> --interpret_method <interpretation method> --outcome_id <outcome index> --out_file <output feature attritbution file>

# Implement the Seal framework for predicting variant effect on expression in 122 brain tissue states (from early fetal to adulthood)
python seal_interpret.py --vcf_file test/predict/test_var.vcf --model_info_file model/tissue_state_early_fetal_to_adult/tissue_state_early_fetal_to_adult_seal_model_summary.txt --interpret_method deeplift --outcome_id 0 --out_file test/interpret/test_var_tissue_state_early_fetal_to_adult 
