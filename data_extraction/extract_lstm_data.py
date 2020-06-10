import numpy as np
import pandas as pd
import spacy as sp

import glob
import os,sys,inspect
import argparse
import re
import ast
import json

from xml.etree import ElementTree

CURRENTDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,'%s/SPPAS'%CURRENTDIR)
sys.path.insert(3,CURRENTDIR)
import SPPAS.sppas.src.anndata.aio.readwrite as spp

from textblob import TextBlob
from collections import Counter

#################### SPACY & IPU
def extract_tier(file_name, tier_name="Transcription"):
    """Read file and extract tier for analysis (either left or right tier)
    
    Input:
    -----------
    file_name: str
        name of file to be analysed
    tier_name: str 
        tiers to look for before using IPUs
    
    Output:
    -----------
    tier: sppasTier
    """
    parser = spp.sppasRW(file_name)
    trs = parser.read()
    if tier_name in [tier.get_name() for tier in trs]: # needs checking since missing tier_name still return sth
        tier = parser.read().find(tier_name)
    else:
        tier = parser.read().find("IPUs")
    return tier

def get_interval(sppasObject):
    """ Return the transcriped text and the IPU start/stop points for a given annotation.
    
    Input:
    -----------
    sppasObject: sppasAnnotation
    
    Output:
    -----------
    label: str
    [start, stop]: array of floats
    [start_radius, stop_radius]: array of floats
    """
    # speech
    label = sppasObject.serialize_labels() # labels => returns speech only
    # location
    location = sppasObject.get_location()[0][0]
    start = location.get_begin().get_midpoint()
    stop = location.get_end().get_midpoint()
    # the radius represents the uncertainty of the localization
    start_radius = location.get_begin().get_radius()
    stop_radius = location.get_end().get_radius()

    return label, [start, stop], [start_radius, stop_radius]

def get_ipu(tier, minimum_length = 0.5):
    """List all IPUs for one tier and returns transcribed text (if not empty) and start/stop points.
    
    Input:
    -----------
    tier: sppasTier
    minimum_length: float
        duration of short answers to remove
    
    Output:
    -----------
    p: pd.DataFrame
    """
    d = []

    for sppasOb in tier:
        label, [start, stop], [start_r, stop_r] = get_interval(sppasOb)
        if label in ["#", "", " ", "***", "*"]:
            continue
        else:
            if (stop - start) >= minimum_length:
                d.append({'label': label, 'start':start, 'stop':stop, 'duration':stop - start})
    return pd.DataFrame(d)

def tag_one(sentence, nlp=sp.load('fr_core_news_sm')):
    """Extract POS tags for one sentence or one - aligning with MarsaTag (simplified output: no Cc nor Cs)
    
    Input:
    --------
    sentence: str
        sentence extracted from Textgrid
    nlp: spacy language model
        default: 'fr_core_news_sm'
    
    Output:
    --------
    p_spacy: pd.DataFrame
        shape [form, pos]
    """
    p_spacy = []
    d_marsa = {'ADJ': 'ADJ', 'ADP':'PREP', 'ADV': 'ADV', 'AUX': 'VERB', 'CONJ':'CONJ', 'CCONJ':'CONJ', \
              'DET':'DET', 'INTJ': 'INTJ', 'NOUN':'NOUN', 'NUM':'DET', 'PART':'ADV', 'PRON':'PRON', \
              'PROPN':'NOUN', 'PUNCT':'PUNCT', 'SCONJ':'CONJ', 'SYM':'X', 'VERB':'VERB', 'X':'X', 'SPACE':'SPACE', '':'ERROR'}
    for d in nlp(sentence):
        try:
            p_spacy.append({'form':d, 'original_pos':d.pos_, 'pos': d_marsa[d.pos_]})
        except:
            print(d, d.pos_)
    return pd.DataFrame(p_spacy)

#################### COMPLEXITY FUNCTIONS
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


#################### READ ALL AND CREATE 
def filename_analyser(fn):
    """
    example: S19_Sess3_CONV2_002-conversant.TextGrid
    """
    [sub, block, conv, nb] = fn.replace('.TextGrid', '').split('_')
    [nb, tier] = nb.split('-')
    return int(sub[1:]), int(block.replace('Sess', '')), int(conv.replace('CONV', '')), int(nb), tier

def folder_analysis(input_folder, neuro_file, minimum_length=0.5):
    """Apply MarsaTag to all .TextGrid files to get XML
    
    Input:
    ---------
    input_folder: str
        local path, Jupyter cannot access files outside of root
    neuro_file: str
        absolute path, path to neuro data
    minimum_length: float
        duration of short answers to remove - for Spacy analysis
    
    Output:
    ---------
    l: list
        list of dictionaries of shape [sub, trial, agent, x = [], y = [list of mean neuro activations]]
    """
    datan = pd.read_csv(neuro_file, sep='\t', header=None, names=["area", "locutor", "session", "image", "bold", "agent", "trial"], skipfooter=1)
    datan.agent = datan.agent.apply(lambda x: x.strip())
    p = []
    for f in sorted(os.listdir(input_folder)):
        if 'participant.TextGrid' in f: # removing .DS_Store and other files
            sub, block, conv, nb, tier = filename_analyser(f)
            # if marsa create output folder and analyse
            fp_in = os.path.join(input_folder, f)
            d = {'locutor': sub, 'trial': (3*(block-1)+(nb-1)//2)+1, 'agent':('H' if conv == 1 else 'R')}
            d['y'] = datan[(datan.locutor == d['locutor']) & (datan.trial == d['trial']) & (datan.agent == d['agent'])].bold.values.tolist()
            
            # load file from .textgrid
            trs = extract_tier(fp_in)
            data = get_ipu(trs, minimum_length)
            # ipu functions
            data['qt_feedback'] = data.label.apply(lambda x: count_terms(x, MAIN_FEEDBACK_ITEMS))
            data['qt_discourse'] = data.label.apply(lambda x: count_terms(x, MAIN_DISCOURSE_ITEMS))
            data['qt_filled_pause'] = data.label.apply(lambda x: count_terms(x, FILLED_PAUSE_ITEMS))
            for col in ['qt_feedback', 'qt_discourse', 'qt_filled_pause']:
                data[col.replace('qt', 'has')] = (data[col] > 1).astype(int)
            data['speech_rate'] = (data.label.apply(lambda x: (count_syllables(x)))/data.duration)
            # add to data
            d['x'] = data[[col for col in data.columns if col != 'label']].values.tolist()
            d['columns'] = [col for col in data.columns if col != 'label']
            p.append(d)
        
    return p

if __name__ == '__main__':
    p = folder_analysis(os.path.join(CURRENTDIR,'convers/head/Transcriptions'), os.path.join(CURRENTDIR,"data_neuro/Full.txt"), minimum_length=0)
    with open(os.path.join(CURRENTDIR,'data/data_lstm.txt'), 'w') as json_file:
        json.dump(p, json_file)