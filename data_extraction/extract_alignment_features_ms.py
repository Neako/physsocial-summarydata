"""
Adapted version of extract_alignment_features for analysis using MarsaTag on IPUs

Execution:
$ python data_extraction/extract_alignment_features_ms.py conversant participant -m 'convers/marsa_split'
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
import copy

try: # if issues with spacy not being installed...
    import spacy as sp
except:
    import warnings
    warnings.warn("spacy not installed")

####### ALIGNMENT FEATURES
remove_lemmas = {'aux':["avoir", "aller", "devoir", "pouvoir", "venir", "vouloir", "savoir", "faire", "falloir"], 
                    'etat':["être", "devenir", "paraître", "sembler", "ressembler", "rester", "apparaître", "tomber", "vivre"] }
# patterns à conserver: NOUN, ADJ, VERB - si lemma pas dans la liste

def create_conversation_ms(folder_conv, folder_part):
    """Same as previous function, except adapted for using MarsaTag on each IPU

    Input:
    ------------
    folder_conv: str
        location of locutor 1
    folder_part: str
        location of locutor 2
    
    Output:
    ------------
    p: pd.DataFrame
        dataframe of shape ['start_time', 'stop_time', 'tier', 'concatenated_text', 'concatenated_data']
    """
    p = []
    #
    for folder,name in zip([folder_conv, folder_part], ["conversant", "participant"]):
        list_df = []
        for file in os.listdir(folder):
            if '.xml' in file:
                [start, end] = file.replace('.xml', '').split('_')
                all_file = os.path.join(folder, file)
                try:
                    df = marsatag_to_pandas(read_marsa(all_file), with_inserted=False)
                    df['lemma'] = df.apply(lambda x: x.lemma if x.lemma is not None else x.form, axis=1) # deal with unknown forms
                    data = df[['form', 'pos', 'lemma']].to_dict('record')
                    list_df.append({'start':float(start), 'stop':float(end), 'label':' '.join(df.form), 'data':data})
                except:
                    # in some cases, marsatag file is empty
                    print("\tEmpty file: "+all_file)
        list_df = pd.DataFrame(list_df)
        list_df['tier'] = name
        p.append(list_df)
    # concatenating and aggregating
    p = pd.concat(p).sort_values(by='start').reset_index(drop=True)
    df = p.groupby((p.tier!=p.tier.shift()).cumsum()).agg(**{'concatenated_text':pd.NamedAgg(column='label', aggfunc= (lambda x: ' '.join(x))),
                                                            'start_time':pd.NamedAgg(column='start', aggfunc= min),
                                                            'stop_time': pd.NamedAgg(column='stop', aggfunc= max),
                                                            'concatenated_data': pd.NamedAgg(column='data', aggfunc= (lambda l: [y for x in l for y in x])) }).reset_index(drop=True)
    add_tier = ('conversant', 'participant') if p.tier[0] == 'conversant' else ('participant', 'conversant')
    df['tier'] = [ add_tier[i%2] for i in range(df.shape[0])]
    return df


def extract_pv_text(df, tier='conversant', using_marsasplit=False):
    """ Extracting, for each line of "tier", what comes before / after for the other locutor

    Input:
    --------
    df: pd.DataFrame
        output of create_conversation, shape ['start_time', 'stop_time', 'tier', 'concatenated_text']
        every line is a different locutor
    tier: str
        in 'conversant', 'participant'
    using_marsasplit: bool
        whether df has additional column ['concatenated_data']
    
    Output:
    --------
    pv_text: list 
        list of dict of shape ['all_before_oth','before_oth', 'tier', 'after_oth'] containing concatenated dialog
    """
    oth = 'participant' if tier == 'conversant' else 'conversant'
    pv_text = []
    for i in list(df[df.tier == tier].index):
        tmp = {'before_oth': '' if i == 0 else ' '.join(df[df.index == i-1].concatenated_text),
                'all_before_oth': ' '.join(df[(df.tier == oth) & (df.index < i)].concatenated_text), 
                'tier': df.concatenated_text[i],
                'after_oth': ' '.join(df[(df.tier == oth) & (df.index >= i)].concatenated_text)}
        if using_marsasplit:
            tmp['all_before_nlp'] = [y for x in df[(df.tier == oth) & (df.index < i)].concatenated_data.to_list() for y in x]
            tmp['before_nlp'] = [] if i == 0 else df[df.index == i-1].concatenated_data.values[0]
            tmp['tier_nlp'] = df.concatenated_data[i]
            tmp['after_nlp'] =[y for x in df[(df.tier == oth) & (df.index >= i)].concatenated_data.to_list() for y in x]
        pv_text.append(tmp)
    return pv_text

# LILLA - duplicate function, ms case
def extract_vocab_count_ms(df, keep_POS = ['NOUN', 'ADJ', 'VERB'], exceptions = [y for x in remove_lemmas.values() for y in x]):
    """ Same as previous, except using MarsaTag instead of spacy

    Input:
    --------
    df: pd.DataFrame or list
        output of extract_pv_text, shape ['before_oth', 'tier', 'after_oth', 'before_nlp', 'tier_nlp', 'after_nlp']
    keep_POS: list
        list of strings, of POS tags to keep for content (lemmas)
    exceptions: list
        list of strings, lemmas of verbs not to count in content words
    
    Output:
    --------
    l: list
        list of lists (one list: 1 sentence - sublist: tuples ('word', 'nb_occurrences'))
    target_vocab: list
        list of unique vocab used by the target -- after 1rst sentence
    introduced_vocab: list
        list of unique vocab used and introduced by the prime
    prime_vocab: list
        list of unique vocab used by the prime -- all vocab
    ignored_vocab+target_vocab: list
        list of unique vocab used by the target -- all vocab
    """
    l = []
    prime_vocab = [] # prime vocab at step t
    introduced_vocab = [] # prime vocab at step t _without words introduced by target_
    ignored_vocab = [] # target 1rst intervention
    target_vocab = [] # target vocab at step t minus 1rst intervention
    for it, d in enumerate(df): # for each conversant sentence
        pre = d['before_nlp']
        post = d['after_nlp']
        # Add vocab matching keep_POS tags to curr_vocab / ignored_vocab (lemmas)
        if it==0:
            ignored_vocab = [ x['lemma'] for x in pre if (x['pos'] in keep_POS) and (x['lemma'] not in exceptions) ]
        else:
            target_vocab += [ x['lemma'] for x in pre if (x['pos'] in keep_POS) and (x['lemma'] not in exceptions) ]
        curr_vocab = [ x['lemma'] for x in d['tier_nlp'] if (x['pos'] in keep_POS) and (x['lemma'] not in exceptions) ]
        prime_vocab += curr_vocab # for analysis
        curr_vocab = [x for x in curr_vocab if (x not in set(ignored_vocab+target_vocab))] # already considering lemmas
        introduced_vocab += curr_vocab
        # 3. Count vocab in post speech
        l.append(Counter([x['lemma'] for x in post if x['lemma'] in curr_vocab]))
    # add last sentences in 'post' to target vocab
    target_vocab += [ x['lemma'] for x in post if (x['pos'] in keep_POS) and (x['lemma'] not in exceptions) ]

    return l, list(set(target_vocab)), list(set(introduced_vocab)), list(set(prime_vocab)), list(set(ignored_vocab+target_vocab))

# no more use for scc

###### WRAP 
def folder_analysis(marsa_folder, primes=['conversant', 'participant'], with_inserted=False):
    """Extract alignment features.

    Disclaimer: yes, some features are very similar (esp. lengths). Comparison repetition/common & spacy/marsatag
    
    Input:
    ---------
    marsa_folder: str
        if not None, path to MarsaTag output folder; if None, Spacy is used instead
    primes: list
        list of str, tier to use as prime
    with_inserted: bool
        whether to add inserted punctuation (MarsaTag) in analysis
    
    Output:
    ---------
    p: pd.DataFrame
        results of complexity functions for each conversation
    s: dict (json)
        extracts of each file
    """
    p = []
    for folder in sorted(os.listdir(marsa_folder)):
        fd_in = os.path.join(marsa_folder, folder)
        if (os.path.isdir(fd_in)) and ('conversant' in folder):
            sub, block, conv, nb, _ = filename_analyser(folder)
            # extract conversation 
            df = create_conversation_ms(fd_in, fd_in.replace('conversant', 'participant'))
            d = {'file': folder.replace('-conversant',''), 
                    'locutor':sub, 'block':block, 'conv': conv, 'it':nb }
            d['Trial'] = 3*(d['block']-1)+(d['it']-1)//2
            try:
                for prime in primes:
                    d['prime'] = prime
                    vc = extract_pv_text(df, tier=prime, using_marsasplit=True)
                    # output is: l, target_vocab, introduced_vocab, prime_vocab, ignored_vocab+target_vocab
                    counters, target_voc, prime_voc, all_prime_voc, all_target_voc = extract_vocab_count_ms(vc)
                    counters = {k:v for c in counters for k,v in c.items()}
                    lilla = len(counters) / (len(prime_voc)*len(target_voc)) # fonction existence 
                    d['lilla'] = lilla
                    d['lilla_num'] = len(counters)
                    d['prime_contentw_l'] = len(prime_voc)
                    d['target_contentw_l'] = len(target_voc)
                    d['prime_contentw'] = ' '.join(prime_voc)
                    d['target_contentw'] = ' '.join(target_voc)
                    d['all_prime_contentw'] = ' '.join(all_prime_voc)
                    d['all_target_contentw'] = ' '.join(all_target_voc)
                    d['repeated_contentw'] = ' '.join(counters.keys())
                    d['target_introducedw'] = ' '.join(set(all_prime_voc) - set(prime_voc))
                    d['target_introducedw_l'] = len(list(set(all_prime_voc) - set(prime_voc)))
                    # appending to df
                    p.append(copy.deepcopy(d)) # shallow copy not enough
            except:
                print("Error with: "+folder)

    return pd.DataFrame(p)


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--marsa_folder', '-m', type=str, default='convers/marsa_split') 
    parser.add_argument('primes', nargs='+', type=str) # ['conversant', 'participant']
    parser.add_argument('--output_file', '-o', type=str, default='data/extracted_align_ms_data.xlsx')
    args = parser.parse_args()
    print(args)
    p = folder_analysis(args.marsa_folder, args.primes)
    
    # reorder columns
    ordered_columns=['file','locutor', 'block', 'conv', 'it', 'Trial', 'prime']
    removed_columns=['data', 'extract_text']
    other_columns=sorted(list(set(p.columns) - set(ordered_columns) - set(removed_columns)))
    # write to file
    p[ordered_columns + other_columns].to_excel(args.output_file, index=False, header=True)