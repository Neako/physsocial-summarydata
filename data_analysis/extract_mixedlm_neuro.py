"""
Original code: _ipynb/test_neuro.ipynb

Execute:
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness -s 'data/' -o 'data/pvalues.xlsx' -i 'data_analysis/_img'
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness -s 'data/' -o 'data/pvalues.xlsx' -e True
$ python data_analysis/extract_mixedlm_neuro.py sum_ipu_lgth lexical_richness speech_rate_min4 -s 'data/' -o 'data/pvalues.xlsx' -i 'data_analysis/_img' -l 'data/extracted_data.xlsx' -n 'data_neuro/Full_stats.xlsx' -r 1 4 19 23

Execute for align:
$ python data_analysis/extract_mixedlm_neuro.py lilla  -s 'data/' -o 'data/pvalues_align.xlsx' -i 'data_analysis/_img' -l 'data/extracted_align_data.xlsx' -n 'data_neuro/Full_stats.xlsx' -r 1 4 19 23 -a True

TODO:
* Insert metadata in JSON files (esp. removed subjects & file taken from)
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
from matplotlib.colors import PowerNorm

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
    results.rename(columns={'conv_id_unif':"Trial"}, inplace=True)
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

def execute_glm(merneuro, int_cols, areas, formula, re_f):
    """Execute GLM

    Input:
    --------
    merneuro: pd.DataFrame
        shape: linguistiq features (_conv, _part and _diff for each features) + brain areas as columns; sessions as rows
    int_cols: list
        list of strings, interest columns name, prgram argument "functions"
    areas: list
        list of areas in the neuro file. extracted before renaming occured
    formula: str
        raw formula for smf.mixedlm()
    re_f: str
        re_formula for smf.mixedlm()
    
    Output:
    --------
    pvalues: dict
        contains models pvalues, shape {'int_col': {'formula': np.array}}
    estimates: dict
        contains models estimates, shape {'int_col': {'formula': np.array}}
    """
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning # logging errors: either ConvergenceWarning or RuntimeWarning
    # saving in
    pvalues = {}
    estimates = {}

    for c in int_cols:
        print(c)
        p_c_dic = {}
        e_c_dic = {}
        for formula_part in ['_part', '_conv', '_diff']:
            int_cols = ['Intercept', c+formula_part] + (['Agent[T.R]', c+formula_part+':Agent[T.R]'] if re.search('Agent', formula) is not None else [])
            start_time = time.time()
            print('\t', formula_part)
            p_f_dic = []
            e_f_dic = []
            for ar in areas:
                formula_1 = formula.format(str(ar).zfill(3), c+formula_part)
                print(formula_1)
                md = smf.mixedlm(formula_1, merneuro, groups=merneuro["locutor"], re_formula=re_f)
                with warnings.catch_warnings(record=True) as w:
                    mdf = md.fit()
                # Add warnings to model data
                p_to_dic = mdf.pvalues[int_cols]
                p_to_dic['Warning'] = None if len(w) == 0 else str(w[-1].category).replace("<class '", '').replace("'>", '').split('.')[-1]
                e_to_dic = mdf.fe_params
                e_to_dic['Warning'] = None if len(w) == 0 else str(w[-1].category).replace("<class '", '').replace("'>", '').split('.')[-1]
                # Add to dic - no need to add "area" bc continuous set of areas, starting at 0 (control)
                p_f_dic.append(p_to_dic)
                e_f_dic.append(e_to_dic)
            p_c_dic[formula_part] = pd.DataFrame(p_f_dic)
            e_c_dic[formula_part] = pd.DataFrame(e_f_dic)
            print("\tElapsed: {0:4.2f}".format(time.time() - start_time))
        pvalues[c] = p_c_dic
        estimates[c] = e_c_dic
    metadata={'pvalues': [c.replace(formula_part, '{}') for c in p_c_dic[formula_part].columns], 
        'estimates': [c.replace(formula_part, '{}') for c in e_c_dic[formula_part].columns]
    }

    return pvalues, estimates, metadata

def saving_as_json(pvalues, estimates, metadata, json_folder, name_insert=''):
    """Save pvalues and estimates to different files in json_folder for later analysis + add metadata
    """
    with open(os.path.join(json_folder,'pvalues'+name_insert+'.txt'), 'w') as json_file:
        data = {c:{f:df.values.tolist() for f, df in v.items()} for c,v in pvalues.items()}
        data['metadata'] = {k:v for k,v in metadata.items()}
        data['metadata']['columns'] = metadata['pvalues']
        json.dump(data, json_file)
    with open(os.path.join(json_folder,'estimates'+name_insert+'.txt'), 'w') as json_file:
        data = {c:{f:df.values.tolist() for f, df in v.items()} for c,v in estimates.items()}
        data['metadata'] = {k:v for k,v in metadata.items()}
        data['metadata']['columns'] = metadata['estimates']
        json.dump(data, json_file)

def loading_as_json(json_folder, name_insert=''):
    with open(os.path.join(json_folder,'pvalues'+name_insert+'.txt'), 'r') as json_file:
        pvalues = json.load(json_file)
    with open(os.path.join(json_folder,'estimates'+name_insert+'.txt'), 'r') as json_file:
        estimates = json.load(json_file)
    return pvalues, estimates

def df_to_excel(pvalues, int_cols, excel_path):
    """Writing results to excel - can be called with pvalues or estimates dictionnary
    """
    writer = pd.ExcelWriter(excel_path)
    for c in int_cols:
        for formula_part in ['_part', '_conv', '_diff']:
            df = pvalues[c][formula_part]
            if not isinstance(df, pd.core.frame.DataFrame):
                # load from dictionary
                cols = pvalues['metadata']['columns']
                df = pd.DataFrame(df, columns=cols)
            df.sort_values(by=c+formula_part, ascending=True, inplace=True) # cannot use interaction as some results won't have that
            df.to_excel(writer, sheet_name=c+formula_part)
    writer.save()
    print('Saved successfully')

def img_to_file(int_cols, formulas, pvalues, img_folder):
    for c in int_cols:
        for formula_part in formulas:
            plt.subplots(figsize=(40, 5))
            df = pvalues[c][formula_part]
            if not isinstance(df, pd.core.frame.DataFrame):
                df = pd.DataFrame(df, columns=['Intercept', 'Agent[T.R]', c+formula_part, c+formula_part+':Agent[T.R]', 'Warning'])
            df['Warning'] = df['Warning'].apply(lambda x: 1 if x is not None else 0)
            sns_plot = sns.heatmap(df.T, norm=PowerNorm(gamma=1./3.), cbar_kws={'ticks': [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]})
            sns_plot.figure.savefig(os.path.join(img_folder, '{}{}.png'.format(c, formula_part)))

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('functions', nargs='+', type=str)
    parser.add_argument('--json_folder', '-s', type=str, default=None)
    parser.add_argument('--linguistic_data', '-l', type=str, default='data/extracted_data.xlsx')
    parser.add_argument('--neuro_data', '-n', type=str, default='data_neuro/Full.txt')
    parser.add_argument('--remove_subjects', '-r', type=int, nargs='+', default=[])
    parser.add_argument('--excel_output', '-o', type=str, default=None)
    parser.add_argument('--json_exists', '-e', type=bool, default=False)
    parser.add_argument('--is_align', '-a', type=bool, default=False)
    parser.add_argument('--img_folder', '-i', type=str, default=None)
    parser.add_argument('--lm_formula', '-form', type=str, default="area_{} ~ {} * Agent + Trial", help="arguments changed in loop should be addressed as {} ex: area_{} or feature = {}")
    parser.add_argument('--re_formula', '-re', type=str, default="1 + Trial")
    args = parser.parse_args()
    print(args)

    if args.is_align:
        print("\nIn the following, _part results are 'participant as prime', _conv results are 'conversant as prime'")
    # case 1: json has not been created
    if not args.json_exists:
        results, datan = load_data(os.path.join(CURRENTDIR, args.linguistic_data), os.path.join(CURRENTDIR, args.neuro_data), neuro_is_csv=(re.search('.xlsx', args.neuro_data) is None), ling_is_align=args.is_align)
        areas = datan.area.unique()
        main_cols = ['locutor', 'Trial', 'Agent']
        merres, merneuro = create_df(results, datan, main_cols, args.functions, which_remove=args.remove_subjects)
        pvalues, estimates, metadata = execute_glm(merneuro, args.functions, areas, formula = args.lm_formula, re_f=args.re_formula)
        # Adding metadata to the file
        metadata['formula'] = args.lm_formula
        metadata['re_f'] = args.re_formula
        metadata['ling_file'] = args.linguistic_data
        metadata['neuro_file'] = args.neuro_data
        if args.json_folder is not None:
            saving_as_json(pvalues, estimates, metadata, os.path.join(CURRENTDIR, args.json_folder), name_insert=('' if not args.is_align else '_align'))

    # case 2: load from json
    if args.json_exists:
        try:
            pvalues, estimates = loading_as_json(os.path.join(CURRENTDIR, args.json_folder), name_insert=('' if not args.is_align else '_align'))
            for (k,v) in pvalues['metadata'].items():
                if k != 'columns':
                    print(k, v)
        except:
            print('JSON files do not exist!')

    # write to img
    if args.img_folder is not None:
        img_to_file(args.functions, ['_part', '_conv', '_diff'], pvalues, args.img_folder)

    # write to file
    if args.excel_output is not None:
        df_to_excel(pvalues, args.functions, os.path.join(CURRENTDIR, args.excel_output))
        df_to_excel(estimates, args.functions, os.path.join(CURRENTDIR, args.excel_output).replace('pvalues', 'estimates'))
