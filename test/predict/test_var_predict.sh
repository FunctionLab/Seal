# !/usr/bin/env bash
## created by Yun Hao @FunctionLab 2023
## This script contains command line examples that implement the Seal framework for variant effect prediction in tissue and cell states of brain development

# Usage:
# python seal_predict.py --vcf_file <variant vcf file> --model_info_file <Seal model summary file> --out_file <Output model prediction file>

# Implement the Seal framework for predicting variant effect on expression in 122 brain tissue states (from early fetal to adulthood)
python seal_predict.py --vcf_file test/predict/test_var.vcf --model_info_file model/tissue_state_early_fetal_to_adult/tissue_state_early_fetal_to_adult_seal_model_summary.txt --out_file test/predict/test_var_tissue_state_effect_pred.tsv > test/predict/test_var_tissue_state_effect_pred.log

# Implement the Seal framework for predicting variant effect on expression in 598 brain cell states (early fetal)
python seal_predict.py --vcf_file test/predict/test_var.vcf --model_info_file model/cell_state_early_fetal/cell_state_early_fetal_seal_model_summary.txt --out_file test/predict/test_var_cell_state_effect_pred1.tsv > test/predict/test_var_cell_state1_effect_pred1.log

# Implement the Seal framework for predicting variant effect on expression in 82 brain cell states (mid fetal to adulthood)
python seal_predict.py --vcf_file test/predict/test_var.vcf --model_info_file model/cell_state_mid_fetal_to_adult/cell_state_mid_fetal_to_adult_seal_model_summary.txt --out_file test/predict/test_var_cell_state_effect_pred2.tsv > test/predict/test_var_cell_state_effect_pred2.log
