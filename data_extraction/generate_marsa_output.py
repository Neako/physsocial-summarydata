""" Generate MarsaTag output files for all Transcriptions
# data folder structure: file_loc.png

Excute:
$ python data_extraction/generate_marsa_output.py

Execute with splitted Textgrids:
$ python data_extraction/generate_marsa_output.py -i convers/transcript_split -o convers/marsa_split -sp True
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
    
    Remove MarsaTag output:
        https://askubuntu.com/questions/98377/how-to-hide-terminal-output-when-executing-a-command

    MarsaTag arguments:
        * -r: format (Textgrid)
        * -pt: tier
        * --oral
        * -P: POS
        * -ix: input format
        * -ox: output format
        * -od: output folder
    
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
    #print(os.popen(s + input_path + ' > ' + output_path).read())
    # better options + 
    os.system(s + " -ix .TextGrid -ox .xml -od {} {} > /dev/null 2>&1".format(output_path, input_path))

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
                    os.makedirs(output_folder)
                if f.replace('.TextGrid', '.xml') not in os.listdir(output_folder):
                    one_marsa(fp_in, output_path=fp_out)
            except:
                print("\tError with file:\t"+f)

def folder_analysis_split(input_folder, output_folder):
    """Apply MarsaTag to all .TextGrid files to get XML
    
    Input:
    ---------
    input_folder: str
        local path
    output_folder: str
    """
    l = [x[0].replace(CURRENTDIR+"/","") for x in os.walk(CURRENTDIR)]
    if not output_folder in l: # os.walk list all files in subfolders
        os.makedirs(output_folder)
    for folder in sorted(os.listdir(input_folder)):
        fd_in = os.path.join(input_folder, folder)
        if os.path.isdir(fd_in): # normal behavior
            fp_md = os.path.join(output_folder, folder)
            print(fp_md)
            if fp_md not in l:
                os.makedirs(fp_md)
            for f in sorted(os.listdir(os.path.join(input_folder, folder))):
                if '.TextGrid' in f: # removing .DS_Store and other files
                    fp_in = os.path.join(input_folder, folder, f)
                    if f.replace('.TextGrid', '.xml') not in os.listdir(fp_md):
                        fp_out = os.path.join(fp_md, f.replace('.TextGrid', '.xml'))
                        one_marsa(fp_in, output_path=fp_md)#fp_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', '-o', type=str, default='convers/marsatag')
    parser.add_argument('--input_folder', '-i', type=str, default='convers/head/Transcriptions')
    parser.add_argument('--split_files', '-sp', type=bool, default=False, help="Whether IPUs are stored in different files & need to go deeper into structure")
    args = parser.parse_args()
    if args.split_files:
        folder_analysis_split(args.input_folder, args.output_folder)
    else:
        folder_analysis(args.input_folder, args.output_folder)