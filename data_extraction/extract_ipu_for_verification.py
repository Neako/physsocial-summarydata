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

def filename_analyser(fn):
    """
    example: S19_Sess3_CONV2_002-conversant.TextGrid
    """
    [sub, block, conv, nb] = fn.replace('.TextGrid', '').split('_')
    [nb, tier] = nb.split('-')
    return int(sub[1:]), int(block.replace('Sess', '')), int(conv.replace('CONV', '')), int(nb), tier

def folder_analysis(input_folder, minimum_length=0.5):
    """Apply MarsaTag to all .TextGrid files to get XML
    
    Input:
    ---------
    input_folder: str
        local path, Jupyter cannot access files outside of root
    minimum_length: float
        duration of short answers to remove - for Spacy analysis
    
    Output:
    ---------
    l: list
        list of dictionaries of shape [sub, trial, agent, x = [], y = [list of mean neuro activations]]
    """
    p = []
    for f in sorted(os.listdir(input_folder)):
        if '.TextGrid' in f: # removing .DS_Store and other files
            sub, block, conv, nb, tier = filename_analyser(f)
            fp_in = os.path.join(input_folder, f)
            
            # load file from .textgrid
            trs = extract_tier(fp_in)
            data = get_ipu(trs, minimum_length)
            data['locutor'] = sub
            data['trial'] = (3*(block-1)+(nb-1)//2)+1
            data['agent'] = ('H' if conv == 1 else 'R')
            data['file_name'] = fp_in
            data['tier'] = tier
            data['speech_rate'] = (data.label.apply(lambda x: (count_syllables(x)))/data.duration)
            p.append(data)
        
    return pd.concat(p)

if __name__ == '__main__':
    p = folder_analysis(os.path.join(CURRENTDIR,'convers/head/Transcriptions'), minimum_length=0)
    p.to_excel(os.path.join(CURRENTDIR,'data/ipu_full.xlsx'), index=False)