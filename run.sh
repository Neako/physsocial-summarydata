#!/bin/bash
# https://openclassrooms.com/fr/courses/43538-reprenez-le-controle-a-laide-de-linux/43126-afficher-et-manipuler-des-variables
basicf_file="data/extracted_data_last.xlsx"
alignf_file="data/extracted_align_ms_data.xlsx"
pvalues_basicf="data/pvalues_last.xlsx"
est_basicf="data/estimates_last.xlsx"
pvalues_alignf="data/pvalues_align_last.xlsx"
est_alignf="data/estimates_align_last.xlsx"
img_folder='data_analysis/_img/extract_last'
extract_marsa="n"

if [ $extract_marsa = "y" ]
then
    # extract marsa all and ipu
    python data_extraction/generate_marsa_output.py -i convers/transcript_split -o convers/marsa_split -sp True -sk 1 4 19 23
    python data_extraction/generate_marsa_output.py -sk 1 4 19 23
fi

#### run for basic features
python data_extraction/extract_basic_features.py lexical_richness linguistic_complexity content_complexity extract_text -m 'convers/marsatag' -o $basicf_file
python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness content_complexity -s 'data/' -o $pvalues_basicf -i 'data_analysis/_img/heatmaps_formula_normal' -l $basicf_file -n 'data_neuro/Full3_stats.xlsx' -r 1 4 19 23
# plotting brain - pvalues & estimates
python data_analysis/visbrain_render.py -d $pvalues_basicf -p 0.001 -i $img_folder -v left right -hem True
python data_analysis/visbrain_render.py -d $est_basicf -p 0.001 -i $img_folder -v left right -hem True -t estimates

#### run for align features
python data_extraction/extract_alignment_features_ms.py conversant participant -m 'convers/marsa_split'
python data_analysis/extract_mixedlm_neuro.py lilla  -s 'data/' -o 'data/pvalues_align.xlsx' -i 'data_analysis/_img/heatmaps_formula_normal' -l $alignf_file -n 'data_neuro/Full3_stats.xlsx' -r 1 4 19 23 -a True
# plotting brain - pvalues & estimates
python data_analysis/visbrain_render.py -d $pvalues_alignf -p 0.001 -i $img_folder -v left right -hem True
python data_analysis/visbrain_render.py -d $est_alignf -p 0.001 -i $img_folder -v left right -hem True -t estimates

#### run analyses - basic
python data_analysis/generate_summary.Rmd.py lexical_richness linguistic_complexity content_complexity sum_ipu_lgth mean_ipu_lgth qt_discourse qt_feedback qt_filled_pause ratio_discourse nratio_feedback ratio_filled_pause speech_rate_min4 -l $basicf_file -r 1 4 19 23 -e "summary_last.xlsx" 
python data_analysis/generate_test.Rmd.py lexical_richness linguistic_complexity content_complexity sum_ipu_lgth mean_ipu_lgth ratio_discourse nratio_feedback ratio_filled_pause -o "basic_features_last.Rmd" -l $basicf_file -p True -r 1 4 19 23 -ee True -eo "summary_last.xlsx"
# run analyses - align
python data_analysis/generate_align_test.Rmd.py lilla -l $alignf_file -p True -r 1 4 19 23 -sm "summary_align_last.xlsx"