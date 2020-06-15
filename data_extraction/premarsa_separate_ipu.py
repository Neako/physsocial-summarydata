"""
Splitting all transcripts to apply MarsaTag to IPUs and _keep_ those separated

Doc for SPPAS: http://www.sppas.org/documentation_06_scripting.html
"""

import numpy as np
import pandas as pd

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

from utils import *

def create_split(trs, parser, annot, file_loc, tier_name="Transcription"):
    """Save the annotation to a separate file

    Input:
    -------
    trs: sppasTranscription object
        must have be emptied of the tier we're reading
    parser: sppasRW parser
    annot: sppasAnnotation object
        annotation we're writing to file
    file_loc: str
        folder name - initial textgrid name
    tier_name: str
        which tier to write to, default "Transcription"
    """
    label, [start, stop], _ = get_interval(annot)
    if label not in ["#", "", " ", "***", "*"]:
        # create new tier and add annotation
        nt = trs.create_tier(tier_name)
        nt.append(annot)
        # add to new file
        parser.set_filename(os.path.join(file_loc, "{0:2.10f}_{1:2.10f}.TextGrid".format(start, stop)))
        parser.write(trs)

if __name__ == '__main__':
    print(CURRENTDIR)
    input_folder = 'convers/head/Transcriptions'
    output_folder = 'convers/transcript_split'
    l = [x[0].replace(CURRENTDIR+"/","") for x in os.walk(CURRENTDIR)] # storing created folders
    if not output_folder in l: 
        os.makedirs(output_folder)
    # looping over files
    for f in sorted(os.listdir(input_folder)):
        if '.TextGrid' in f: # removing .DS_Store and other files
            o_folder = os.path.join(output_folder, f.replace('.TextGrid', ''))
            if not o_folder in l: 
                os.makedirs(o_folder)
                # then create files
                fp_in = os.path.join(input_folder, f)
                tier, trs, parser = extract_tier(fp_in, return_parser=True)
                for annot in tier:
                    # remove previous tier from trs - since we only use this object
                    try: # not all annotations are written to file so there might be some errors here
                        trs.pop([t.get_name() for t in trs].index('Transcription'))
                    except:
                        pass
                    # write in sep files
                    create_split(trs, parser, annot, o_folder)
            else:
                # whether to consider everything is awesooooome and nothing needs to be done
                # def yes not doing anything.
                continue

