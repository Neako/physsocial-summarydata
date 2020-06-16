import numpy as np
import pandas as pd
try: # if issues with spacy not being installed...
    import spacy as sp
except:
    import warnings
    warnings.warn("spacy not installed")

import glob
import os,sys,inspect
import argparse
import re
import ast

import matplotlib.pyplot as plt
from xml.etree import ElementTree

CURRENTDIR = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,'%s/SPPAS'%CURRENTDIR)
sys.path.insert(3,CURRENTDIR)
import SPPAS.sppas.src.anndata.aio.readwrite as spp

from textblob import TextBlob
from collections import Counter
import seaborn as sns


########### TEXTGRID FUNCTIONS
def extract_tier(file_name, tier_name="Transcription", return_parser=False):
    """Read file and extract tier for analysis (either left or right tier)
    
    Input:
    -----------
    file_name: str
        name of file to be analysed
    tier_name: str 
        tiers to look for before using IPUs
    return_parser: bool
        whether to return IO parser and sppasTranscription object, default False
    
    Output:
    -----------
    tier: sppasTier
    trs: sppasTranscription object
        if return_parser is True
    parser: IO parser
        if return_parser is True
    """
    parser = spp.sppasRW(file_name)
    trs = parser.read()
    if tier_name in [tier.get_name() for tier in trs]: # needs checking since missing tier_name still return sth
        tier = parser.read().find(tier_name)
    else:
        tier = parser.read().find("IPUs")
    if return_parser:
        return tier, trs, parser
    else:
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


###### SPACY FUNCTIONS
def tag_one(sentence, nlp):
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
              'PROPN':'NOUN', 'PUNCT':'PUNCT', 'SCONJ':'CONJ', 'SYM':'X', 'VERB':'VERB', 'X':'X', 'SPACE':'SPACE'}
    for d in nlp(sentence):
        try:
            p_spacy.append({'form':d, 'original_pos':d.pos_, 'pos': d_marsa[d.pos_]})
        except:
            print(d, d.pos_)
    return pd.DataFrame(p_spacy)

###### MARSATAG FUNCTIONS
def read_marsa(file_path):
    """Read XML file and return list of tokens
    
    Known structure:
    ---------
    <?xml version='1.0' encoding='UTF-8'?>
    <?xml-stylesheet type="text/xsl" href="mars.xsl"?>
    <document format="MARS_1.0">
        <sample>
        
        ... List of tokens:
        <token form="..." regex_type="..." features="...">
            ... One solution and several propositions:
            <solution msc=".." type="..."/>
            <proposition msc=".." frequency=".."/>
        </token>
        
        </sample>
    </document>
    
    Input:
    ---------
    file_path: str
        local path, Jupyter cannot access files outside of root, XML file
    
    Output:
    ---------
    tree: xml.etree.ElementTree.Element
    """
    tree = ElementTree.parse(file_path)
    # getroot() gets document, getchildren()[0] gets sample ==> access tokens
    return tree.getroot().getchildren()[0] 

def marsatag_to_sentence(document, remove_punct=False, remove_ipu=True):
    sentence = ''
    for child in document:
        if child.tag == 'token':
            sentence += '' if (len(sentence) == 0 or child.attrib['form'] in [',', '.', "'"]) else ' '
            sentence += child.attrib['form']
    return sentence

def marsatag_to_pandas(document, with_inserted=True):
    """Select words & tags from xml 
    
    Input:
    -------
    document: xml.etree.ElementTree.Element
    with_inserted: bool
        whether to remove punctuation inserted by MarsaTag
    
    Output:
    -------
    sentence: dataframe
        shape ['form', 'pos', 'lemma', 'inserted']
    """
    sentence = []
    d = {'A': 'ADJ', 'D':'DET', 'R': 'ADV', 'V': 'VERB', 'C': 'CONJ', 'N': 'NOUN', 
         'S':'PREP', 'W':'PUNCT', 'I':'INTJ', 'P':'PRON', 'U':'X'}
    # exceptions:
    exceptions = ['batman', 'spiderman'] # complete with others if need be
    # also exception: all words with 'regex_type="not_in_the_lexicon"' and containing '$' ==> 'NOUN'
    for child in document:
        if child.tag == 'token':
            # first child is solution, if exists
            try:
                # for all words like $Avengers$ $Disney$ $Furhat$...
                if (child.attrib['regex_type'] == "not_in_the_lexicon") and ('$' in child.attrib['form']):
                    sentence.append({'form': child.attrib['form'].replace('$','').lower(), \
                                'pos': 'NOUN', \
                                'lemma': child.attrib['form'].replace('$','').lower(), \
                                'inserted': (child.attrib['regex_type'] == 'inserted') \
                                })
                # for other references that aren't in $...$
                elif (child.attrib['regex_type'] == "not_in_the_lexicon") and (child.attrib['form'] in exceptions):
                    sentence.append({'form': child.attrib['form'], \
                                'pos': 'NOUN', \
                                'lemma': child.attrib['form'].replace('$','').lower(), \
                                'inserted': (child.attrib['regex_type'] == 'inserted') \
                                })
                # basically every other word - won't deal with unknown words that aren't entities
                else: 
                    sentence.append({'form': child.attrib['form'], \
                                'pos': d[child.attrib['features'][0]], \
                                'lemma': None if 'lemma' not in child.attrib.keys() else child.attrib['lemma'], \
                                'inserted': (child.attrib['regex_type'] == 'inserted') \
                                })
            except: # errors like """<token form="-" regex_type="Ponct_Wm1">"""
                sentence.append({'form': child.attrib['form'], \
                             'pos': 'PUNCT', \
                             'lemma': None if 'lemma' not in child.attrib.keys() else child.attrib['lemma'], \
                             'inserted': (child.attrib['regex_type'] == 'inserted') \
                            })
    if with_inserted:
        return pd.DataFrame(sentence)
    else:
        p = pd.DataFrame(sentence)
        return p[~p.inserted]

###### OTHER RECURRENT FUNCTIONS
# cleaning text
patterns_dic = {' [sS]pider[. -]{0,1}man ': ' Spiderman ', 
                ' [bB]at[. -]{0,1}man ': ' Batman ', ' \[(.+?)\] ':' ', ' (\w+-|-\w+) ':' ', "' ":"'"}

def clean_text(text, patterns_dic):
    # remove '$'
    t = text.replace('$', '')
    # replace patterns
    for k,v in patterns_dic.items():
        t = re.sub(k,v,t)
    return t

def extract_text(df, replace_in_text=True, join_with=' '):
    if replace_in_text:
        return clean_text(' '.join(df.form.values), patterns_dic)
    else:
        return ' '.join(df.form.values)

# folder analysis
def filename_analyser(fn):
    """
    example: S19_Sess3_CONV2_002-conversant.TextGrid
    """
    [sub, block, conv, nb] = fn.replace('.TextGrid', '').split('_')
    [nb, tier] = nb.split('-')
    return int(sub[1:]), int(block.replace('Sess', '')), int(conv.replace('CONV', '')), int(nb), tier