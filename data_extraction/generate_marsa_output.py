""" Generate MarsaTag output files for all Transcriptions
# data folder structure: file_loc.png

Excute:
$ python data_extraction/generate_marsa_output.py
"""

import numpy as np
import pandas as pd

import glob
import os,sys,inspect
import argparse
import re
import ast
from xml.etree import ElementTree
from collections import Counter

CURRENTDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,'%s/SPPAS'%CURRENTDIR)
sys.path.insert(3,CURRENTDIR)

def one_marsa(input_path, locutor = 'Transcription', output_path='log.xml'):
    """Execute one parsing of MarsaTag with CLI command, saves as XML
    
    MarsaTag arguments:
        * -r: format (Textgrid)
        * -pt: tier
        * --oral
        * -P: POS
    
    Input:
    ---------
    input_path: str
        local path, Jupyter cannot access files outside of root
    locutor: str
        tiers to look for before using IPUs
    output_path: str
        local path, XML file, default: log.xml
    """
    s = "./MarsaTag/MarsaTag-UI.sh --cli -r praat-textgrid -pt {} -P --oral ".format(locutor)
    #os.system(s + input_path + ' > ' + output_path)
    print(os.popen(s + input_path + ' > ' + output_path).read())

def folder_analysis(input_folder, output_folder):
    """Apply MarsaTag to all .TextGrid files to get XML
    
    Input:
    ---------
    input_folder: str
        local path
    output_folder: str
    """
    l = [x[0].replace(CURRENTDIR+"/","") for x in os.walk(CURRENTDIR)]
    for f in sorted(os.listdir(input_folder)):
        if '.TextGrid' in f: # removing .DS_Store and other files
            fp_in = os.path.join(input_folder, f)
            try:
                fp_out = os.path.join(output_folder, f.replace('.TextGrid', '.xml'))
                if not output_folder in l: # os.walk list all files in subfolders
                    print('here', CURRENTDIR)
                    os.makedirs(output_folder)
                if f.replace('.TextGrid', '.xml') not in os.listdir(output_folder):
                    one_marsa(fp_in, output_path=fp_out)
            except:
                print("\tError with file:\t"+f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', '-o', type=str, default='convers/marsatag')
    parser.add_argument('--input_folder', '-i', type=str, default='convers/head/Transcriptions')
    args = parser.parse_args()
    folder_analysis(args.input_folder, args.output_folder)