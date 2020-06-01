"""
Python file for generating neuro images

See https://github.com/EtienneCmb/visbrain/blob/master/visbrain/objects/scene_obj.py

Execute:
$ python data_analysis/visbrain_render.py -d data/pvalues.xlsx -f sum_ipu_lgth -c part -on interaction -p 0.001 -i 'data_analysis/_img'
$ python data_analysis/visbrain_render.py -d data/pvalues.xlsx -p 0.001 -i 'data_analysis/_img' -v left right

TODO:
* Add parameter to take value into account
* Add parameter to take pvalue into account when plotting estimate
"""
import pandas as pd
import numpy as np

import sys
import os
import argparse

from visbrain.gui import Brain
from visbrain.objects import BrainObj, SceneObj
import nibabel as nib

CURRENTDIR = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')

def read_data(data:str, parcellation:str, feature:str, on:str, pmax:float):
    """
    Input:
    -------
    feature: str
        concatenation of 'feature' and 'condition' args (for instance, sum_ipu_lgth_part)
    on: str
        from args, in ['agent', 'feature', 'interaction']
    pmax: float
        maximum pvalue, for area selection
    """
    # neuro files
    l_file = os.path.join(CURRENTDIR, os.path.join(args.parcellation, "lh.BN_Atlas.annot"))
    r_file = os.path.join(CURRENTDIR, os.path.join(args.parcellation, "rh.BN_Atlas.annot"))
    #annot_l = nib.freesurfer.io.read_annot(l_file)
    #annot_r = nib.freesurfer.io.read_annot(r_file)
    # stats
    d_on = {'agent':'Agent[T.R]', 'feature':feature, 'interaction':feature+':Agent[T.R]'}
    brain_areas = pd.read_excel(os.path.join(CURRENTDIR, args.data), sheet_name=feature, index_col=0)
    # .reset_index() # after resetting, areas are in "index"
    brain_areas_select = list(brain_areas[brain_areas[d_on[on]] <= pmax].index)
    # brain_areas_select = pd.DataFrame({"Name": l, "Label":l, "Activation":[1]*len(l)})
    # return
    return l_file, r_file, brain_areas_select

def create_brain_obj(l_file, r_file, activated_areas, brain_name):
    """Create BrainObj, add areas, and return object

    Output:
    --------
    b_obj: visbrain.objects.BrainObj
    """
    b_obj = BrainObj(brain_name, translucent=False, hemisphere='both')
    annot_data = b_obj.get_parcellates(l_file)
    
    # errors if missing labels - removing labels not in annot_data
    activated_areas = list(set(activated_areas) - (set(activated_areas) - set(annot_data.index)))
    select = annot_data['Labels'][activated_areas].tolist()
    left_areas = [x for x in select if x[-1] == 'L']
    right_areas = [x for x in select if x[-1] == 'R']

    if len(left_areas) > 0:
        b_obj.parcellize(l_file, hemisphere='left',  select=left_areas, data=np.random.shuffle(np.arange(len(left_areas))), cmap='rainbow')
    if len(right_areas) > 0:
        b_obj.parcellize(r_file, hemisphere='right',  select=right_areas, data=np.random.shuffle(np.arange(len(right_areas))), cmap='rainbow')
    return b_obj

def visualise_brain(b_obj):
    """Opens Visbrain GUI with colored ROI
    """
    vb = Brain(brain_obj=b_obj, bgcolor='black')
    vb.show()

def generate_img(l_file, r_file, activated_areas, brain_name, out_file, views):
    """Generate .png and .gif animation of rotating brains
    """
    sc = SceneObj(size=(1500, 1000))
    KW = dict(title_size=14., zoom=2.)
    # PLOT OBJECTS
    for i,rot in enumerate(views):
        # cannot use the same object
        sc.add_to_subplot(create_brain_obj(l_file, r_file, activated_areas, brain_name), row=i//2, col=i%2, rotate=rot, title=rot, **KW)
    # gif and png
    # sc.preview()
    sc.record_animation(out_file + "_areas.gif")
    sc.screenshot(saveas = out_file + "_areas.png")
    return sc.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str) # xlsx file
    parser.add_argument('--feature', '-f', type=str, default=None) # which sheet? ex: sum_ipu_lgth
    parser.add_argument('--condition', '-c', choices=['part', 'conv', 'diff'], default=None)
    parser.add_argument('--on', '-on', choices=['agent', 'feature', 'interaction'], default='agent') # Agent[T.R]
    parser.add_argument('--pmax', '-p', type=float, default=0.01) # pvalue
    parser.add_argument('--parcellation', '-n', type=str, default='data_neuro/parcellation') 
    parser.add_argument('--img_folder', '-i', type=str, default='data_neuro/parcellation')
    parser.add_argument('--brain', '-b', choices=['white', 'inflated'], default='white')
    parser.add_argument('--views', '-v', nargs='+', choices=['right', 'left', 'top', 'bottom', 'front', 'back'], default=['right', 'left', 'top', 'bottom', 'front', 'back'])
    args = parser.parse_args()
    print(args)
    
    if args.feature is None:
        xl = pd.ExcelFile(args.data)
        features = list(xl.sheet_names)
    else:
        features = [args.feature+'_'+args.condition]
    ons = ['agent', 'feature', 'interaction'] if args.on is None else [args.on]

    for f in features:
        for on in ons:
            print('\n{} {}'.format(f, on))
            # read files
            l_file, r_file, brain_areas_select = read_data(args.data, args.parcellation, f, on, args.pmax)
            print("Activated Areas: ")
            print(brain_areas_select)
            b_obj = create_brain_obj(l_file, r_file, brain_areas_select, args.brain)

            if args.img_folder is None:
                visualise_brain(b_obj)
            else:
                generate_img(l_file, r_file, brain_areas_select, args.brain, os.path.join(CURRENTDIR,os.path.join(args.img_folder, f+'_'+on+'_'+str(args.pmax))),args.views)
    
