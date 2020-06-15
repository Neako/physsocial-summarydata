"""
Extracting
* sum_ipu_lgth, mean_ipu_lgth, ratio_silence_lgth
* polarity / subjectivity
* complexities
* feedbacks, discourse, filled_pause

Execution:
$ python data_extraction/extract_basic_features.py lexical_richness linguistic_complexity extract_text extract_sentiment extract_subjectivity content_complexity -m 'convers/marsatag'
"""

from utils import *

import numpy as np
import pandas as pd

import glob
import os,sys,inspect
import argparse
import re
import ast
from collections import Counter

import spacy as sp
from textblob import TextBlob


###### COMPLEXITY FUNCTIONS
def lexical_richness(df):
    return df[(df.pos == 'ADV') | (df.pos == 'ADJ')].shape[0] / df.shape[0]

def linguistic_complexity(df):
    return df[(df.pos == 'CONJ') | (df.pos == 'PREP') | (df.pos == 'PRON')].shape[0] / df.shape[0]

def content_complexity(df):
    remove_lemmas = {'aux':["avoir", "aller", "devoir", "pouvoir", "venir", "vouloir", "savoir", "faire", "falloir"], 
                    'etat':["être", "devenir", "paraître", "sembler", "ressembler", "rester", "apparaître", "tomber", "vivre"] }
    exceptions = [y for x in remove_lemmas.values() for y in x]
    return df[(df.pos == 'NOUN') | (df.pos == 'ADJ') | ((df.pos == 'VERB') & (~df.lemma.isin(exceptions)))].shape[0] / df.shape[0]

def ipu_length(df):
    # Cannot be used with MarsaTag Output
    return df.duration.mean()

def parse_sentiment(s:str):
    # https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
    blob = TextBlob(s)
    return blob.sentiment

def extract_sentiment(df, replace_in_text=True, join_with=' '):
    return parse_sentiment(extract_text(df, replace_in_text=True, join_with=' '))[0]

def extract_subjectivity(df, replace_in_text=True, join_with=' '):
    return parse_sentiment(extract_text(df, replace_in_text=True, join_with=' '))[1]

# discourse
OK_FORMS = [u"o.k.",u"okay",u"ok",u"OK",u"O.K."]
VOILA_FORMS = [u"voilà",u"voila"]
DACCORD_FORMS = [u"d'accord",u"d' accord"]
LAUGHTER_FORMS = [u'@',u'@ @',u'@@']
EMO_FORMS = [u'@',u'@ @',u'@@',u'ah',u'oh']

REGULATORY_DM_SET = set([u"mh",u"ouais",u"oui",u"o.k.",u"okay",u"ok",u"OK",u"O.K.",u"d'accord",u"voilà",u"voila",u'bon',u"d'",
u"accord",u'@',u'@ @',u'@@',u'non',u"ah",u"euh",u'ben',u"et",u"mais",u"*",u"heu",u"hum",u"donc",u"+",u"eh",u"beh",u"donc",u"oh",u"pff",u"hein"])

FILLED_PAUSE_ITEMS = [u"euh",u"heu",u"hum",u"mh"]
SILENCE = [u'+',u'#',u'',u'*']
LAUGHTER = [u'@',u'@@']

MAIN_FEEDBACK_ITEMS = [u"mh",u"ouais",u"oui",u'non',u'ah',u"mouais"]+ OK_FORMS + VOILA_FORMS + DACCORD_FORMS + LAUGHTER_FORMS
MAIN_DISCOURSE_ITEMS = [u"alors",u"mais",u"donc",u'et',u'puis',u'enfin',u'parceque',u'parcequ',u'ensuite']
MAIN_PARTICLES_ITEMS = [u"quoi",u"hein",u"bon",u'mais',u'ben',u'beh',u'enfin',u'vois',u'putain',u'bref']

def count_terms(s:str, item_list:list):
    q = 0
    for it in item_list:
        q += len(re.findall(it, s))
    return q

# syllables
def count_syllables(s):
    """Counting syllables (a good approximation for syllables is the number of vowels phonemes in the text)
    Rules:
    * replacing qu --> q (since 'u' is silent)
    * removing final 'e' and plural 'es' (basically words longer than 4)

    Input:
    -----------
    s: str
        text to parse
    Output:
    -----------
    len: int
        number of vowels/syllables
    """
    vowels = ['ouai','eui','uei','yeu','oeu','eau','uoi','oui','aie','eoi','eai','ea','eo','eâ','uê','uî', 'ui','eû','oî','oû','oi','ué','où','io','ie','ue','oy','ai','eu','ei','au','ou','ée','ë','ü','ï','â','ô','ê','î','û','è','à','ù','é','y','i','a','e','o','u']
    s1 = s.replace('qu','q')

    rules_list = []
    for wd in s1.split():
        if len(wd) <= 2:
            rules_list.append(wd)
        elif wd[-1] == 'e':
            rules_list.append(wd[:-1])
        elif wd[-2:] == 'es' and len(wd) > 4:
            rules_list.append(wd[:-2])
        else:
            rules_list.append(wd)
    rules_list = ' '.join(rules_list)

    # replacing all vowels for easier count - starting with rarest/longest phonemes 
    for vw in vowels:
        rules_list = rules_list.replace(vw, "XXX")

    return len(re.findall('XXX',rules_list))


###### WRAP ALL
def extract_stats(p_pos, cplx_functions, to_pandas=True):
    """Analyse with all functions
    
    Input:
    ----------
    p: pd.DataFrame
        shape [word, pos_tag], extracted either from Spacy or MarsaTag
    cplx_functions: array
        list of complexity functions to use on the interaction
    to_pandas: bool
        whether to return data as dataframe or list of dict
        
    Output:
    ----------
    d: pd.Series
        shape [function_name, result]
    """
    d = {}
    for f in cplx_functions:
        d[f.__name__] = f(p_pos)
    if to_pandas:
        return pd.Series(d)
    else:
        return d

def folder_analysis(input_folder, marsa_folder, cplx_functions, 
                    ipu_analysis=False, quantity_analysis=False,
                    speech_rate=False, with_inserted=False, minimum_length=0.5):
    """Apply MarsaTag to all .TextGrid files to get XML
    
    Input:
    ---------
    input_folder: str
        local path, Jupyter cannot access files outside of root
    marsa_folder: str
        if not None, path to MarsaTag output folder; if None, Spacy is used instead
    cplx_functions: array
        list of complexity functions to use on the interaction
    ipu_analysis: bool
        whether to use spacy for IPUs analysis or not, default False
    quantity_analysis: bool
        whether to compute quantity metrics 
        count_terms(s, MAIN_FEEDBACK_ITEMS), 
        count_terms(s, MAIN_DISCOURSE_ITEMS), 
        count_terms(s, FILLED_PAUSE_ITEMS)
    speech_rate: bool
        whether to add counting of syllables and speech rate - based on extracted text
    with_inserted: bool
        whether to add inserted punctuation (MarsaTag) in analysis
    minimum_length: float
        duration of short answers to remove - for Spacy analysis
    
    Output:
    ---------
    p: pd.DataFrame
        results of complexity functions for each conversation
    s: dict (json)
        extracts of each file
    """
    p = []
    s = {}
    for f in sorted(os.listdir(input_folder)):
        if '.TextGrid' in f: # removing .DS_Store and other files
            sub, block, conv, nb, tier = filename_analyser(f)
            # if marsa create output folder and analyse
            fp_in = os.path.join(input_folder, f)
            d = {}
            if marsa_folder is not None:
                try:
                    document = read_marsa(os.path.join(marsa_folder, f.replace('.TextGrid', '.xml')))
                    p_analysis = marsatag_to_pandas(document, with_inserted=with_inserted)
                except:
                    print("Error with file:\t" + f)
            # if spacy or ipu analysis load file from .textgrid
            if (marsa_folder is None) or (ipu_analysis) or (quantity_analysis):
                trs = extract_tier(fp_in)
                data = get_ipu(trs, minimum_length)
                if (marsa_folder is None):
                    try:
                        p_analysis = tag_one('. '.join(data.label.values), nlp=sp.load('fr_core_news_sm'))
                    except:
                        print("Error with file:\t" + f)
                        
            d = extract_stats(p_analysis, cplx_functions, to_pandas=False)
            # add to data
            d['file'] = f
            d['locutor'] = sub
            d['tier'] = tier
            d['block'] = block
            d['conv'] = conv
            d['it'] = nb
            d['conv_id_unif'] = 3*(d['block']-1)+(d['it']-1)//2
            d['nb_tokens'] = p_analysis.shape[0]
            # add ipu_analysis to d # potential issue empty file
            if ipu_analysis and ('duration' in data.columns):
                d['mean_ipu_lgth'] = data.duration.mean() 
                d['sum_ipu_lgth'] = data.duration.sum()
                d['ratio_silence_lgth'] = (59-d['sum_ipu_lgth'])/59
            if quantity_analysis:
                sp = ' '.join(data.label.values)
                d['qt_feedback'] = count_terms(sp, MAIN_FEEDBACK_ITEMS)
                d['qt_discourse'] = count_terms(sp, MAIN_DISCOURSE_ITEMS)
                d['qt_filled_pause'] = count_terms(sp, FILLED_PAUSE_ITEMS)
                for col in ['qt_discourse', 'qt_feedback', 'qt_filled_pause']:
                    d[col.replace('qt', 'ratio')] = d[col] / d['nb_tokens']
                d['nratio_feedback'] = data.label.apply(lambda x: int(count_terms(x.split(' ')[0], MAIN_FEEDBACK_ITEMS) > 0)).sum() / data.shape[0]
                d['nratio_discourse'] = data.label.apply(lambda x: int(count_terms(x, MAIN_DISCOURSE_ITEMS) > 0)).sum() / data.shape[0]
                d['nratio_filled_pause'] = data.label.apply(lambda x: int(count_terms(x, FILLED_PAUSE_ITEMS) > 0)).sum() / data.shape[0]
            if speech_rate:
                d['count_syllables'] = count_syllables(d['extract_text'])
                d['speech_rate'] = d['count_syllables']/d['sum_ipu_lgth']
                if ipu_analysis:
                    d['nb_ipu'] = data.shape[0]
                    # removing laugther only ipus
                    j = re.compile('[aeiouéèêàûùôïyAEIOUY]')
                    #d['speech_rate_mean'] = (data.label.apply(lambda x: (count_syllables(x)))/data.duration).mean()
                    #d['speech_rate_min'] = (data.label.apply(lambda x: (count_syllables(x)))/data.duration).min()
                    #d['speech_rate_max'] = (data.label.apply(lambda x: (count_syllables(x)))/data.duration).max()
                    d['speech_rate_mean'] = (data.apply(lambda x: (count_syllables(x.label))/x.duration if re.search(j,x.label) is not None else None, axis=1)).mean()
                    d['speech_rate_min'] = (data.apply(lambda x: (count_syllables(x.label))/x.duration if re.search(j,x.label) is not None else None, axis=1)).min()
                    d['speech_rate_max'] = (data.apply(lambda x: (count_syllables(x.label))/x.duration if re.search(j,x.label) is not None else None, axis=1)).max()
                    d['speech_rate_2'] = (data.label.apply(lambda x: count_syllables(x) if re.search(j,x) is not None else None)).sum()/(data.apply(lambda x: x.duration if re.search(j,x.label) is not None else None, axis=1)).sum()
                    # tmp
                    tmp = (data.apply(lambda x: x.duration if count_syllables(x.label) >= 4 else None, axis=1)).sum()
                    d['speech_rate_min4'] = (data.label.apply(lambda x: count_syllables(x) if count_syllables(x) >= 4 else None)).sum()/tmp if tmp > 0 else 0
            p.append(d)
            d['data'] = p_analysis
            s[f] = d
        
    return pd.DataFrame(p), s


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--marsa_folder', '-m', type=str, default=None) # if None, Spacy is used; otherwise MarsaTag
    parser.add_argument('--convers_folder', '-i', type=str, default='convers/head/Transcriptions')
    parser.add_argument('--has_ipu_analysis', '-ip', type=bool, default=True) # whether to add textgrid parsing
    parser.add_argument('--has_quantity_analysis', '-q', type=bool, default=True) # whether to count discourse elements
    parser.add_argument('--has_speech_rate', '-sr', type=bool, default=True) # whether to add speech_rate
    parser.add_argument('--minimum_length', '-ml', type=float, default=0.) # for textgrid analysis
    parser.add_argument('complexity_functions', nargs='+', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='data/extracted_data.xlsx')
    args = parser.parse_args()
    functions = [globals()[f] for f in args.complexity_functions] # matching *actual* functions to named passed as args
    p, s = folder_analysis(args.convers_folder, args.marsa_folder, functions,
                    args.has_ipu_analysis, args.has_quantity_analysis,
                    args.has_speech_rate, False, args.minimum_length)
    # reorder columns
    ordered_columns=['file','locutor', 'block', 'conv', 'it', 'conv_id_unif', 'tier']
    removed_columns=['data', 'extract_text']
    other_columns=sorted(list(set(p.columns) - set(ordered_columns) - set(removed_columns)))
    # write to file
    p[ordered_columns + other_columns + ['extract_text']].to_excel(args.output_file, index=False, header=True)