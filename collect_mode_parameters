# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:00:40 2015

@author: gajendrakatuwal
"""

import pandas as pd
#import numpy as np
import os
import glob
import re
#import matplotlib.pyplot as plt
import pickle

os.getcwd()
# os.chdir('../python')

mode_par = '/Volumes/HD1/hpcdarwin/ABIDE/FIRST/mode_par'

def bvars2modepars(bvars_file):
    # extracts mode parameters from txt version of bvars file 
    line =  open(bvars_file,"r").readlines()[1] # mode parameters are in 2nd line
    mode_pars = [x for x in line.split()]
    mode_pars.pop(0) # remove the first word which is the file path
    mode_pars = [float(x) for x in mode_pars]
    return(mode_pars)

def modepars_sub_cortical(sub_cortical, no_modes = None):
    print(sub_cortical)
    # extracts mode parameters for one sub-cortical structure of all the subjects 
    files = glob.glob(os.path.join(mode_par,'*'+sub_cortical+'*.txt')) # files with 
    # the specific sub_cortical structure
    if no_modes is None:
        mode_pars = [bvars2modepars(f) for f in files]
    else:
        mode_pars = [bvars2modepars(f)[:no_modes[sub_cortical]] for f in files]
        
    df = pd.DataFrame(mode_pars)
    subj_id = [re.findall("A000\d+",x)[0] for x in files] # index names are subject ids extracted from filenames
    df.index = subj_id
    return(df)

def save_xls(list_dfs, xls_path, sheet_names = None):
    # saves a list of data frames in an excel file
    import pandas as pd
    writer  =  pd.ExcelWriter(xls_path)
    for n, df in enumerate(list_dfs):
         if sheet_names is not None:
                df.to_excel(writer,sheet_names[n])
         else:
                df.to_excel(writer,'sheet%s' % n)
    writer.save()


sub_cortical_structures = ["BrStem","L_Accu","R_Accu","L_Amyg","R_Amyg","L_Caud",
                         "R_Caud","L_Hipp","R_Hipp","L_Pall","R_Pall","L_Puta",
                         "R_Puta","L_Thal","R_Thal"]
no_modes = [40,50,50,50,50,30,30,30,30,40,40,40,40,40,40]# no. of modes retained
no_modes = pd.Series(data = no_modes,index = sub_cortical_structures) # 

list_dfs = [modepars_sub_cortical(sub_cortical,no_modes = no_modes) for sub_cortical in sub_cortical_structures]

# write the list of data frames in an excel file
# takes really long time
save_xls(list_dfs,'mode_parameters.xlsx',sheet_names = sub_cortical_structures)

#df = pd.concat(list_dfs,keys = sub_cortical_structures) # rbind
df = pd.concat(list_dfs,keys = sub_cortical_structures,axis = 1) # cbind

df.to_csv('mode_parameters.csv')
pickle.dump(df, open( "mode_parameters.p", "wb" ) )


