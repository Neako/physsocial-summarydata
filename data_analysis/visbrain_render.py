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
from visbrain.objects import BrainObj, SceneObj, ColorbarObj
import nibabel as nib
from vispy.visuals import transforms

CURRENTDIR = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')

def read_data(data:str, parcellation:str, feature:str, on:str, pmax:float, is_estimate=False):
    """
    Input:
    -------
    feature: str
        concatenation of 'feature' and 'condition' args (for instance, sum_ipu_lgth_part)
    on: str
        from args, in ['agent', 'feature', 'interaction']
    pmax: float
        maximum pvalue, for area selection

    Output:
    -------
    l_file: str
    r_file: str
    brain_areas: pd.DataFrame
        dataframe of selected areas, shape ['index', 'values']
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
    if is_estimate:
        brain_areas_select = pd.read_excel(os.path.join(CURRENTDIR, args.data).replace('estimates', 'pvalues'), sheet_name=feature, index_col=0)
        brain_areas_select = list(brain_areas_select[brain_areas_select[d_on[on]] <= pmax].index)
        brain_areas = brain_areas[d_on[on]][brain_areas_select].reset_index().rename(columns={d_on[on]:'values'})
    else:
        brain_areas = brain_areas[brain_areas[d_on[on]] <= pmax][d_on[on]].reset_index().rename(columns={d_on[on]:'values'})
    # return
    return l_file, r_file, brain_areas

def create_brain_obj(l_file, r_file, activated_areas, brain_name, hemisphere='both', cmap='copper', vmin=0., vmax=0.01):
    """Create BrainObj, add areas, and return object

    Input:
    --------
    l_file: str
    r_file: str
    brain_areas: pd.DataFrame
        dataframe of selected areas, shape ['index', 'values']
    brain_name: str
        parameter for the BrainObj, in ('white', 'inflated')
    hemisphere: str
        parameter for the BrainObj, in ('both', 'left', 'right)
    cmap: str
        maplotlib colomap to use. For pvalues, 'copper' should be used; for estimates, 'coolwarm'.
    clim: tuple
        colorbar limits. For pvalues, (0,pmax) should be used; for estimates, a symetrical interval.
    
    Output:
    --------
    b_obj: visbrain.objects.BrainObj
    """
    b_obj = BrainObj(brain_name, translucent=False, hemisphere=hemisphere)
    annot_data = b_obj.get_parcellates(l_file)
    
    # errors if missing labels - removing labels not in annot_data
    annot_select = pd.merge(annot_data.reset_index()[['index', 'Labels']], activated_areas, on=['index'], validate="one_to_one")
    annot_select['is_left'] = annot_select['Labels'].apply(lambda x: x[-1] == 'L')

    if annot_select[annot_select.is_left].shape[0] > 0:
        b_obj.parcellize(l_file, hemisphere='left',  select=annot_select[annot_select.is_left]['Labels'].tolist(), data=annot_select[annot_select.is_left]['values'].tolist(), cmap=cmap, vmin=vmin, vmax=vmax, clim=(vmin, vmax))
    if annot_select[~annot_select.is_left].shape[0] > 0:
        b_obj.parcellize(r_file, hemisphere='right',  select=annot_select[~annot_select.is_left]['Labels'].tolist(), data=annot_select[~annot_select.is_left]['values'].tolist(), cmap=cmap, vmin=vmin, vmax=vmax, clim=(vmin, vmax))
    return b_obj

def visualise_brain(b_obj):
    """Opens Visbrain GUI with colored ROI
    """
    vb = Brain(brain_obj=b_obj, bgcolor='black')
    vb.show()

def generate_img(l_file, r_file, activated_areas, brain_name, out_file, views, add_hemispheres, is_estimate=False, cmap='copper', vmin=0., vmax=0.01, save_gif=False):
    """Generate .png and .gif animation of rotating brains
    """
    sc = SceneObj(size=(1000*(len(views)//2 + (1 if add_hemispheres else 0)), 1000*(len(views)>1)))
    KW = dict(title_size=14., zoom=2.) # zoom not working
    CBAR_STATE = dict(cbtxtsz=12, txtsz=10., width=.1, cbtxtsh=3., rect=(-.3, -2., 1., 4.))
    # PLOT OBJECTS
    for i,rot in enumerate(views):
        # cannot use the same object
        b_obj = create_brain_obj(l_file, r_file, activated_areas, brain_name, cmap=cmap, vmin=vmin, vmax=vmax)
        sc.add_to_subplot(b_obj, row=i//2, col=i%2, rotate=rot, title=rot, **KW)
        # Get the colorbar of the brain object and add it to the scene
        # Identical brain ==> same colorbar
    if add_hemispheres:
        # add left brain
        b_obj = create_brain_obj(l_file, r_file, activated_areas, brain_name, hemisphere='right', cmap=cmap, vmin=vmin, vmax=vmax)
        sc.add_to_subplot(b_obj, row=i//2+1, col=0, rotate='left', title='right half', **KW)
        b_obj = create_brain_obj(l_file, r_file, activated_areas, brain_name, hemisphere='left', cmap=cmap, vmin=vmin, vmax=vmax)
        sc.add_to_subplot(b_obj, row=i//2+1, col=1, rotate='right', title='left half', **KW)
    if is_estimate:
        # cmap needs to be set for all objects
        cb_parr = ColorbarObj(b_obj, cblabel='Data to parcellates', **CBAR_STATE)
        # not working properly and can't find a way to rotate that bar
        # sc.add_to_subplot(cb_parr, row=0, col=2, row_span=i//2+1, width_max=200) 
    # gif and png
    # sc.preview()
    if save_gif:
        sc.record_animation(out_file + ('_est' if is_estimate else '') + "_areas.gif")
    sc.screenshot(saveas = out_file + ('_est' if is_estimate else '') + "_areas.png")
    return sc.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str) # xlsx file
    parser.add_argument('--file_type', '-t', choices=['pvalues', 'estimates'], type=str, default='pvalues') # if estimate, lookup pvalues
    parser.add_argument('--feature', '-f', type=str, default=None) # which sheet? ex: sum_ipu_lgth
    parser.add_argument('--condition', '-c', choices=['part', 'conv', 'diff'], default=None)
    parser.add_argument('--on', '-on', choices=['agent', 'feature', 'interaction'], default=None) # Agent[T.R]
    parser.add_argument('--pmax', '-p', type=float, default=0.01) # pvalue
    parser.add_argument('--parcellation', '-n', type=str, default='data_neuro/parcellation') 
    parser.add_argument('--img_folder', '-i', type=str, default=None)
    parser.add_argument('--brain', '-b', choices=['white', 'inflated'], default='white')
    parser.add_argument('--save_gif', '-g', type=bool, default=False)
    parser.add_argument('--views', '-v', nargs='+', choices=['right', 'left', 'top', 'bottom', 'front', 'back'], default=['right', 'left', 'top', 'bottom', 'front', 'back'])
    parser.add_argument('--add_hemispheres', '-hem', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    
    if args.feature is None:
        xl = pd.ExcelFile(args.data)
        features = list(xl.sheet_names)
    else:
        features = [args.feature+'_'+args.condition]
    ons = ['agent', 'feature', 'interaction'] if args.on is None else [args.on]
    is_estimate=(args.file_type == 'estimates')

    for f in features:
        for on in ons:
            print('\n{} {}'.format(f, on))
            # read files
            l_file, r_file, brain_areas = read_data(args.data, args.parcellation, f, on, args.pmax, is_estimate=is_estimate)
            print("Activated Areas: ")
            print(brain_areas['index'].tolist())
            print()

            cmap = 'copper' if not is_estimate else 'coolwarm'
            v = max(abs(brain_areas['values'].min()), brain_areas['values'].max())
            vmin = -1*v if is_estimate else 0
            vmax = v if is_estimate else args.pmax
            print(cmap, vmin, vmax)
            if args.img_folder is None:
                b_obj = create_brain_obj(l_file, r_file, brain_areas, args.brain, cmap=cmap, vmin=vmin, vmax=vmax)
                visualise_brain(b_obj)
            else:
                generate_img(l_file, r_file, brain_areas, args.brain, os.path.join(CURRENTDIR,os.path.join(args.img_folder, f+'_'+on+'_'+str(args.pmax))), args.views, args.add_hemispheres, is_estimate=is_estimate, cmap=cmap, vmin=vmin, vmax=vmax, save_gif=args.save_gif)
    
