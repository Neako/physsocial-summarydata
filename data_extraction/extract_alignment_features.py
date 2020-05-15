"""
Execution:
$ python data_extraction/extract_alignment_features.py conversant participant -m 'convers/marsatag'
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

import spacy as sp

####### ALIGNMENT FEATURES
remove_lemmas = {'aux':["avoir", "aller", "devoir", "pouvoir", "venir", "vouloir", "savoir", "faire", "falloir"], 
                    'etat':["être", "devenir", "paraître", "sembler", "ressembler", "rester", "apparaître", "tomber", "vivre"] }
# patterns à conserver: NOUN, ADJ, VERB - si lemma pas dans la liste

def create_conversation(file_conv, file_part, minimum_length=0., remove_laughter=True):
    """
    Doc for sequential groupby: https://stackoverflow.com/questions/53091343/pandas-groupby-sequential-values

    Input:
    ------------
    file_conv: str
        location of locutor 1
    file_part: str
        location of locutor 2
    minimum_length: float
        duration of short answers to remove - for Spacy analysis
    remove_laughter: bool
        whether to remove laughter (and by extension, ipus comprised of laugther only)
    
    Output:
    ------------
    p: pd.DataFrame
        dataframe of shape ['start_time', 'stop_time', 'tier', 'concatenated_text']
    """
    # file to data 
    p = []
    for f,n in zip([file_conv, file_part], ["conversant", "participant"]):
        tier_conv = extract_tier(f)
        data_conv = get_ipu(tier_conv, minimum_length)
        data_conv["tier"] = n
        p.append(data_conv.drop(np.where(data_conv.label == '@')[0])) # removing laughters - feedback => not removed
    p = pd.concat(p).sort_values(by='start').reset_index(drop=True)
    p.label = p.label.apply(lambda x: x.replace('@', ''))
    df = p.groupby((p.tier!=p.tier.shift()).cumsum()).agg(**{'concatenated_text':pd.NamedAgg(column='label', aggfunc= (lambda x: ' '.join(x))),
                                                            'start_time':pd.NamedAgg(column='start', aggfunc= min),
                                                            'stop_time': pd.NamedAgg(column='stop', aggfunc= max) }).reset_index(drop=True)
    add_tier = ('conversant', 'participant') if p.tier[0] == 'conversant' else ('participant', 'conversant')
    df['tier'] = [ add_tier[i%2] for i in range(df.shape[0])]
    return df

def extract_pv_text(df, tier='conversant'):
    """ Extracting, for each line of "tier", what comes before / after for the other locutor

    Input:
    --------
    df: pd.DataFrame
        output of create_conversation, shape ['start_time', 'stop_time', 'tier', 'concatenated_text']
        every line is a different locutor
    tier: str
        in 'conversant', 'participant'
    
    Output:
    --------
    pv_text: list 
        list of dict of shape ['before_oth', 'tier', 'after_oth'] containing concatenated dialog
    """
    oth = 'participant' if tier == 'conversant' else 'conversant'
    pv_text = []
    for i in list(df[df.tier == tier].index):
        pv_text.append({'before_oth': ' '.join(df[(df.tier == oth) & (df.index < i)][:i].concatenated_text), 
                        'tier': df.concatenated_text[i],
                        'after_oth': ' '.join(df[(df.tier == oth) & (df.index >= i)].concatenated_text)})
    return pv_text

def extract_vocab_count(df, nlp = sp.load('fr_core_news_sm'), keep_POS = ['NOUN', 'ADJ', 'VERB'], exceptions = [y for x in remove_lemmas.values() for y in x]):
    """ Extracts vocabulary introduced by locutor, and counts occurences in posterior dialog by other participant

    Input:
    --------
    df: pd.DataFrame or list
        output of extract_pv_text, shape ['before_oth', 'tier', 'after_oth']
    nlp: Spacy.nlp
        language POS parser
    keep_POS: list
        list of strings, of POS tags to keep for content (lemmas)
    exceptions: list
        list of strings, lemmas of verbs not to count in content words
    
    Output:
    --------
    l: list
        list of lists (one list: 1 sentence - sublist: tuples ('word', 'nb_occurrences'))
    ignored_vocab: list
        list of vocab introducted by the other locutor
    """
    l = []
    tier_vocab = []
    ignored_vocab = []
    d_marsa = {'ADJ': 'ADJ', 'ADP':'PREP', 'ADV': 'ADV', 'AUX': 'VERB', 'CONJ':'CONJ', 'CCONJ':'CONJ', \
              'DET':'DET', 'INTJ': 'INTJ', 'NOUN':'NOUN', 'NUM':'DET', 'PART':'ADV', 'PRON':'PRON', \
              'PROPN':'NOUN', 'PUNCT':'PUNCT', 'SCONJ':'CONJ', 'SYM':'X', 'VERB':'VERB', 'X':'X', 'SPACE':'SPACE', '':'KEYERROR'}
    for d in df: # for each conversant sentence
        # 1. Apply POS tagging - nlp - on every sentence.
        pre = nlp(d['before_oth'])
        post = nlp(d['after_oth'])
        # 2. Add vocab matching keep_POS tags to curr_vocab / ignored_vocab (lemmas)
        ignored_vocab += [ x.lemma_ for x in pre if (d_marsa[x.pos_] in keep_POS) and (x.lemma_ not in exceptions) ]
        curr_vocab = [ x.lemma_ for x in nlp(d['tier']) if (d_marsa[x.pos_] in keep_POS) and (x.lemma_ not in exceptions) and (x.lemma_ not in set(ignored_vocab))]
        tier_vocab += curr_vocab # for analysis
        # 3. Count vocab in post speech
        l.append(Counter([x.lemma_ for x in post if x.lemma_ in curr_vocab]))

    return l, list(set(ignored_vocab)), list(set(tier_vocab))


def compute_scc_voc(prime_text, target_text, keep_POS = None, n=10, exceptions=[y for x in remove_lemmas.values() for y in x]):
    """Compute Spearman's correlation coefficient based on the Xu & Reiter paper.

    Some questions though:
    * comment tu choisis les mots ? est-ce que c'est les mots communs ou les mots les plus fréquents indépendamment ?
    * si c'est indépendamment, comment tu gères qu'un mot n'apparaisse pas chez l'autre ?
    * est-ce que tu prends n'importe quel mot ou n'importe quel mot de contenu ?
    ==> choosing only most frequent words _appearing in both texts_.

    Input:
    --------
    conversant_text: pd.DataFrame
        output of MarsaTag, shape ['form', 'pos', 'lemma', 'inserted']
    participant_text: pd.DataFrame
        output of MarsaTag, shape ['form', 'pos', 'lemma', 'inserted']
    keep_POS: list
        default None, most often ['NOUN', 'ADJ', 'VERB'], vocab to rank
    exceptions: list
        list of words not to take into account
    """
    scc = 0
    if keep_POS is not None:
        prime_text = prime_text[prime_text.pos.isin(keep_POS)]
        target_text = target_text[target_text.pos.isin(keep_POS)]
    # defaults: sort = True, ascending = False / transforming to DF to apply rank for computation
    most_frequent_words = list(pd.concat([prime_text, target_text]).lemma.value_counts().index)[:n]
    prime_rank = prime_text.lemma.value_counts().to_frame()
    prime_rank['rk'] = prime_rank.rank(method="min", ascending=False)
    target_rank = target_text.lemma.value_counts().to_frame()
    target_rank['rk'] = target_rank.rank(method="min", ascending=False)
    # computing
    for word in list(set(most_frequent_words) - set(exceptions)):
        if (word not in prime_rank.index):
            scc += (prime_rank.shape[0] - target_rank.rk[word])
        elif (word not in target_rank.index):
            scc += (prime_rank.rk[word] - target_rank.shape[0])**2
        else:
            scc += (prime_rank.rk[word] - target_rank.rk[word])**2
    # compute scc
    return 1 - 6*scc/(n*(n**2-1))

###### WRAP 
def folder_analysis(input_folder, marsa_folder, primes=['conversant', 'participant'],
                    with_inserted=False, minimum_length=0.5, remove_laughter=True):
    """Extract alignment features.
    
    Input:
    ---------
    input_folder: str
        local path, Jupyter cannot access files outside of root
    marsa_folder: str
        if not None, path to MarsaTag output folder; if None, Spacy is used instead
    with_inserted: bool
        whether to add inserted punctuation (MarsaTag) in analysis
    minimum_length: float
        duration of short answers to remove - for Spacy analysis
    remove_laughter: bool
        whether to ignore laughter in conversation
    
    Output:
    ---------
    p: pd.DataFrame
        results of complexity functions for each conversation
    s: dict (json)
        extracts of each file
    """
    p = []
    for f in sorted(os.listdir(input_folder)):
        if 'conversant.TextGrid' in f: # removing .DS_Store and other files
            sub, block, conv, nb, _ = filename_analyser(f)
            # extract conversation 
            fp_in = os.path.join(input_folder, f) # file full path
            df = create_conversation(fp_in, fp_in.replace('conversant', 'participant'), minimum_length=0., remove_laughter=remove_laughter)
            d = {'file':f.replace('-conversant.TextGrid',''), 'locutor':sub, 'block':block, 'conv': conv, 'it':nb }
            d['Trial'] = 3*(d['block']-1)+(d['it']-1)//2
            try:
                # for scc
                anl = {}
                anl['participant'] = marsatag_to_pandas(read_marsa(os.path.join(marsa_folder, f.replace('conversant.TextGrid', 'participant.xml'))), with_inserted=with_inserted)
                anl['conversant'] = marsatag_to_pandas(read_marsa(os.path.join(marsa_folder, f.replace('.TextGrid', '.xml'))), with_inserted=with_inserted)

                for prime in primes:
                    d['prime'] = prime
                    # LILLA - add window?
                    vc = extract_pv_text(df, tier=prime)
                    counters, prime_voc, target_voc = extract_vocab_count(vc)
                    counters = {k:v for c in counters for k,v in c.items()}
                    lilla = len(counters) / (len(prime_voc)*len(target_voc)) # fonction existence 
                    d['lilla'] = lilla
                    d['prime_contentw'] = len(prime_voc)
                    d['target_contentw'] = len(target_voc)
                    d['log_lilla'] = np.log(lilla)
                    d['midlog_lilla'] = len(counters) / np.log(len(prime_voc)*len(target_voc))
                    # SILLA
                    #counters, prime_syn, target_syn = extract_syntax_count(vc)
                    #counters = {k:v for c in counters for k,v in c.items()}
                    #silla = len(counters) / (len(prime_voc)*len(target_voc)) # fonction existence 
                    #d['silla'] = silla

                    # RepetitionDecay
                    # SCC
                    other = 'participant' if prime == 'conversant' else 'conversant'
                    scc_lex = compute_scc_voc(anl[prime], anl[other], keep_POS = ['NOUN', 'ADJ', 'VERB'], n=10)
                    d['scc_lex'] = scc_lex
                    
                    # appending to df
                    p.append(copy.deepcopy(d)) # shallow copy not enough
            except:
                print(f)

    return pd.DataFrame(p)


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--marsa_folder', '-m', type=str, default=None) # if None, Spacy is used; otherwise MarsaTag
    parser.add_argument('--convers_folder', '-i', type=str, default='convers/head/Transcriptions')
    parser.add_argument('--minimum_length', '-ml', type=float, default=0.) # for textgrid analysis
    parser.add_argument('primes', nargs='+', type=str) # ['conversant', 'participant']
    parser.add_argument('--output_file', '-o', type=str, default='data/extracted_align_data.xlsx')
    args = parser.parse_args()
    print(args)
    p = folder_analysis(args.convers_folder, args.marsa_folder, args.primes, minimum_length=args.minimum_length)
    
    # reorder columns
    ordered_columns=['file','locutor', 'block', 'conv', 'it', 'Trial', 'prime']
    removed_columns=['data', 'extract_text']
    other_columns=sorted(list(set(p.columns) - set(ordered_columns) - set(removed_columns)))
    # write to file
    p[ordered_columns + other_columns].to_excel(args.output_file, index=False, header=True)