"""
Original code: _ipynb/test_neuro.ipynb

Execute:
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness -s 'data/' -o 'data/pvalues.xlsx' -i 'data_analysis/_img'
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness -s 'data/' -o 'data/pvalues.xlsx' -e True
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness speech_rate_min4 -s 'data/' -o 'data/pvalues.xlsx' -i 'data_analysis/_img' -l 'data/extracted_data.xlsx' -n 'data_neuro/Full_stats.xlsx' -r 1 4 19 23

Execute for align:
$ python data_analysis/extract_mixedlm_neuro.py lilla  -s 'data/' -o 'data/pvalues_align.xlsx' -i 'data_analysis/_img' -l 'data/extracted_align_data.xlsx' -n 'data_neuro/Full_stats.xlsx' -r 1 4 19 23 -a True
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels. api as sm
import statsmodels.formula.api as smf
import sys, os
import seaborn as sns
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import ast
import re
import time
import json
import argparse

CURRENTDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def load_data(ling, neuro, neuro_is_csv=False, ling_is_align=False):
    """Loading data as pandas DataFrame from files (location strings) + uniformisation operations

    Input:
    -------
    ling: str
        path from repo root to file
    neuro: str
        path from repo root to file
    neuro_is_csv: bool
        different versions of data have different features / column names, default False: data is xlsx
    ling_is_align: bool
        whether to adapt the data bc of a different source, default False
    
    Output:
    -------
    results: pd.DataFrame
        dataframe of linguistic features [locutor, tier, Trial, Agent]+[interest features]
    datan: pd.DataFrame
        dataframe of neuro features, shape [locutor, session, area, image, bold, Agent, Trial]
    """
    # features
    results = pd.read_excel(os.path.join(CURRENTDIR,ling))
    results.rename(columns={'conv_id_unif':"Trial"})
    if ling_is_align:
        results.rename(columns={'prime':"tier"}, inplace=True) # for treatment purposes only. 
    results['Agent'] = results.conv.apply(lambda x: 'H' if x == 1 else 'R')
    # neuro + operations
    if neuro_is_csv:
        datan = pd.read_csv(neuro, sep='\t', header=None, names=["area", "locutor", "session", "image", "bold", "Agent", "Trial"], skipfooter=1)
        datan.Trial = datan.Trial - 1
        datan.Agent = datan.Agent.apply(lambda x: x.strip()) # remove extra space
    else: # xlsx
        datan = pd.read_excel(os.path.join(CURRENTDIR,neuro))
        datan.rename(columns={col:col.lower() for col in datan.columns}, inplace=True)
        datan.rename(columns={'subj':'locutor', 'sess':'session', 'roi': 'area', 'signal':'bold', 'stat':'bold', 'agent':'Agent', 'idtrial':'Trial'}, inplace=True)
        datan['area'] = datan.area -1
        datan.Agent = datan.Agent.apply(lambda x: x.strip()) # remove extra space
        datan.Trial = datan.Trial - 1
        datan['image'] = 0 # useless either way
    # return
    return results, datan

def create_df(results, datan, main_cols, int_cols, which_remove=[]):
    """Reshape loaded data for GLM + remove problematic subjects

    Output:
    -------
    merres: pd.DataFrame
        temporary dataframe with participant / conversant features side by side
        in case of align data: participant is "participant as prime", conv is "conversant as prime"
    merneuro: pd.DataFrame
        final dataframe containing all features as columns: all brain areas and all features as conv/participant
    """
    participant = results[results.tier == 'participant'][main_cols+int_cols]
    conversant = results[results.tier == 'conversant'][main_cols+int_cols]
    # create pivot data
    datan.area = datan.area.apply(lambda x: 'area_'+str(x).zfill(3))
    pivot_datan = pd.pivot_table(datan, columns='area', values='bold', index=['locutor', 'session', 'Agent', 'Trial', 'image'], aggfunc=np.sum).reset_index()
    # create merge data
    merres = pd.merge(participant, conversant, on=main_cols, suffixes=('_part', '_conv'), validate="one_to_one")
    merneuro = pd.merge(merres, pivot_datan, on=main_cols, suffixes=('_ling', '_bold'), validate="one_to_one")
    merneuro = merneuro[~merneuro.locutor.isin(which_remove)]
    # add diff_columns
    for c in int_cols:
        merneuro[c+'_diff'] = merneuro[c+'_part'] - merneuro[c+'_conv']
    # return 
    return merres, merneuro

def execute_glm(merneuro, int_cols, areas):
    """Execute GLM

    Input:
    --------
    merneuro: pd.DataFrame
        shape: linguistiq features (_conv, _part and _diff for each features) + brain areas as columns; sessions as rows
    int_cols: list
        list of strings, interest columns name, prgram argument "functions"
    areas: list
        list of areas in the neuro file. extracted before renaming occured
    
    Output:
    --------
    pvalues: dict
        contains models pvalues, shape {'int_col': {'formula': np.array}}
    estimates: dict
        contains models estimates, shape {'int_col': {'formula': np.array}}
    """
    pvalues = {}
    estimates = {}

    re_f = "1 + Trial"
    for c in int_cols:
        print(c)
        p_c_dic = {}
        e_c_dic = {}
        for formula_part in ['_part', '_conv', '_diff']:
            start_time = time.time()
            print('\t', formula_part)
            p_f_dic = []
            e_f_dic = []
            for ar in areas:
                formula_1 = "area_{} ~ {} * Agent + Trial".format(str(ar).zfill(3), c+formula_part)
                print(formula_1)
                md = smf.mixedlm(formula_1, merneuro, groups=merneuro["locutor"], re_formula=re_f)
                mdf = md.fit()
                p_f_dic.append(mdf.pvalues[['Intercept', 'Agent[T.R]', c+formula_part, c+formula_part+':Agent[T.R]']])
                e_f_dic.append(mdf.fe_params)
            p_c_dic[formula_part] = pd.DataFrame(p_f_dic)
            e_c_dic[formula_part] = pd.DataFrame(e_f_dic)
            print("\tElapsed: {0:4.2f}".format(time.time() - start_time))
        pvalues[c] = p_c_dic
        estimates[c] = e_c_dic

    return pvalues, estimates

def saving_as_json(pvalues, estimates, json_folder):
    """Save pvalues and estimates to different files in json_folder for later analysis
    """
    with open(os.path.join(json_folder,'pvalues.txt'), 'w') as json_file:
        json.dump({c:{f:df.values.tolist() for f, df in v.items()} for c,v in pvalues.items()}, json_file)
    with open(os.path.join(json_folder,'estimates.txt'), 'w') as json_file:
        json.dump({c:{f:df.values.tolist() for f, df in v.items()} for c,v in estimates.items()}, json_file)

def loading_as_json(json_folder):
    with open('pvalues.txt', 'r') as json_file:
        pvalues = json.load(json_file)
    with open('estimates.txt', 'r') as json_file:
        estimates = json.load(json_file)
    return pvalues, estimates

def df_to_excel(pvalues, int_cols, excel_path):
    """Writing results to excel - can be called with pvalues or estimates dictionnary
    """
    writer = pd.ExcelWriter('pvalues_neuro_nofilter.xlsx')
    for c in int_cols:
        for formula_part in ['_part', '_conv', '_diff']:
            df = pvalues[c][formula_part].sort_values(by=c+formula_part+':Agent[T.R]', ascending=True)
            df.to_excel(writer, sheet_name=c+formula_part)
    writer.save()
    print('Saved successfully')

def img_to_file(int_cols, formulas, pvalues, img_folder):
    for f in int_cols:
        for form in formulas:
            plt.subplots(figsize=(30, 5))
            sns_plot = sns.heatmap(pvalues[f][form].T)
            sns_plot.savefig(os.path.join(img_folder, '{}_{}.png'.format(f, form)))

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('functions', nargs='+', type=str)
    parser.add_argument('--json_folder', '-s', type=str, default=None) # if None, Spacy is used; otherwise MarsaTag
    parser.add_argument('--linguistic_data', '-l', type=str, default='data/extracted_data.xlsx')
    parser.add_argument('--neuro_data', '-n', type=str, default='data_neuro/Full.txt')
    parser.add_argument('--remove_subjects', '-r', type=int, nargs='+', default=[])
    parser.add_argument('--excel_output', '-o', type=str, default=None)
    parser.add_argument('--json_exists', '-e', type=bool, default=False)
    parser.add_argument('--is_align', '-a', type=bool, default=False)
    parser.add_argument('--img_folder', '-i', type=str, default=None)
    args = parser.parse_args()
    print(args)

    if args.is_align:
        print("\nIn the following, _part results are 'participant as prime', _conv results are 'conversant as prime'")
    # case 1: json has not been created
    results, datan = load_data(os.path.join(CURRENTDIR, args.linguistic_data), os.path.join(CURRENTDIR, args.neuro_data), neuro_is_csv=(re.search('.xlsx', args.neuro_data) is None), ling_is_align=args.is_align)
    areas = datan.area.unique()
    main_cols = ['locutor', 'Trial', 'Agent']
    merres, merneuro = create_df(results, datan, main_cols, args.functions, which_remove=args.remove_subjects)
    pvalues, estimates = execute_glm(merneuro, args.functions, areas)
    if args.json_folder is not None:
        saving_as_json(pvalues, estimates, os.path.join(CURRENTDIR, args.json_folder))

    # case 2: load from json
    if args.json_exists:
        try:
            pvalues, estimates = loading_as_json(os.path.join(CURRENTDIR, args.json_folder))
        except:
            print('JSON files do not exist!')

    # write to img
    if args.img_folder is not None:
        img_to_file(args.functions, ['_part', '_conv', '_diff'], pvalues, args.img_folder)

    # write to file
    if args.excel_output is not None:
        df_to_excel(pvalues, args.functions, os.path.join(CURRENTDIR, args.excel_output))
        # df_to_excel(estimates, args.functions, os.path.join(CURRENTDIR, args.excel_output))
