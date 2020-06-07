"""
Generate one-shot R exploration of lilla bricks
Mostly identical to generate_align_test.Rmd.py but with some specific interests (since features fixed)
+ merging align and non align data

Execution:
$ python data_analysis/_exploration/explore_lilla.Rmd.py
"""
import numpy as np
import pandas as pd

import glob
import os,sys,inspect
import argparse
import re
import ast
from collections import Counter
import inspect
import copy

def add_header():
    return """---
title: "R Notebook"
output:
html_document:
    df_print: paged
---
"""

def add_libraries():
    s = """```{r}
library(readxl)
library(sjPlot)
library(ggplot2)
library(lme4)
library(stringr)
library(ggExtra)
```\n"""
    return s

def add_data(linguistic_file=None, remove_subjects=[], is_align=True):
    s = """
```{r}
# linguistic data
data <- read_excel('"""+linguistic_file+"""')
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[!(data$locutor %in% c("""+','.join([str(x) for x in remove_subjects])+""")),]
# data = data[which(data$locutor > 1),]
"""
    if is_align:
        s +="""
data$Trial2 = paste0('t', str_pad(data$Trial, 2, pad = "0"))
data_convprime = data[which(data$prime == "conversant"),]
# data_partprime = data[which(data$prime == "participant"),]
# since we don't look at lilla (no directionaliy needed) both are the same

# Adding new features:
data_convprime$len_cl_p = data_convprime$len_cl
data_convprime$len_cl_t = data_convprime$len_cl
data_convprime$len_cl_over_p = data_convprime$len_cl / data_convprime$len_pl
data_convprime$len_cl_over_t = data_convprime$len_cl / data_convprime$len_tl
```\n
"""
    else: # not align
        s += """
data$Trial = data$conv_id_unif
data$Trial2 = paste0('t', str_pad(data$conv_id_unif, 2, pad = "0"))
```\n
"""
    return s

features = []

def add_mergedata():
    s = """
```{r}
# creating merged data - ling
temp1 = subset(data, select = c("locutor", "conv_id_unif", "Agent", 'sum_ipu_lgth'), tier=='conversant')
colnames(temp1) = c("locutor", "Trial", "Agent", "data_conv")
temp2 = subset(data, select = c("locutor", "conv_id_unif", "Agent", 'sum_ipu_lgth'), tier=='participant')
colnames(temp2) = c("locutor", "Trial", "Agent", "data_part")
merres = merge(temp1, temp2, by=c("locutor", "Trial", "Agent"))
merfull = merge(merres, data_convprime, by=c("locutor", "Trial", "Agent"))
# separating data again
data_r = merfull[which(merfull$Agent == "R"),]
data_h = merfull[which(merfull$Agent == "H"),]
```\n
"""
    return s

def print_saver(excel_output):
    return """
# Saver
```{r}
# Write the first data set in a new workbook
write.xlsx(df_overall, file = '"""+excel_output+"""',
      sheetName = "summary", append = FALSE)
```
"""

def add_saver(functions):
    features = functions + [f+'_over_sumipu' for f in functions]
    s = """
```{r}
# extra columns will add themselves automatically - just creating structures
df_overall = data.frame(mean_h=numeric("""+str(len(features))+"""), 
                std_h=numeric("""+str(len(features))+"""), 
                mean_r=numeric("""+str(len(features))+"""), 
                std_r=numeric("""+str(len(features))+"""), 
                wilcox.pvalue=numeric("""+str(len(features))+"""), 
                row.names = c('"""+"','".join(features)+"""'),
                stringsAsFactors=FALSE)
```
"""
    return s

def add_description(function_name, sum_ipu_length='data_part'):
    s = """
```{r}
# computing values
"""
    for part in ['_r', '_h']:
        s += """
df_overall['"""+function_name+"""', 'mean"""+part+"""'] = mean(data"""+part+"""$'"""+function_name+"""')
df_overall['"""+function_name+"""', 'std"""+part+"""'] = sd(data"""+part+"""$'"""+function_name+"""')

data"""+part+"""$'"""+function_name+"""_over_sumipu' = data"""+part+"""$'"""+function_name+"""'/data"""+part+"""$'"""+sum_ipu_length+"""'
df_overall['"""+function_name+"""_over_sumipu', 'mean"""+part+"""'] = mean(data"""+part+"""$'"""+function_name+"""_over_sumipu')
df_overall['"""+function_name+"""_over_sumipu', 'std"""+part+"""'] = sd(data"""+part+"""$'"""+function_name+"""_over_sumipu')
"""
    s+= """
df_overall['"""+function_name+"""', 'wilcox.pvalue'] = wilcox.test(data_h$'"""+function_name+"""', data_r$'"""+function_name+"""')$p.value
df_overall['"""+function_name+"""_over_sumipu', 'wilcox.pvalue'] = wilcox.test(data_h$'"""+function_name+"""_over_sumipu', data_r$'"""+function_name+"""_over_sumipu')$p.value
```\n
"""
    return s

def create_file(filename, basic_path, align_path, plot_distrib, remove_subjects, excel_output=None):
    # read path to create file
    currdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join(os.path.join(currdir,'data_analysis/_exploration'), filename), 'w')

    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    basic_path = None if basic_path is None else os.path.join(currdir,basic_path)
    align_path = None if align_path is None else os.path.join(currdir,align_path)
    # Adding data
    rmd.write(add_data(align_path, remove_subjects))
    rmd.write(add_data(basic_path, remove_subjects, is_align=False))
    # Merging data
    rmd.write(add_mergedata())
    # Adding dataframe to save data
    # Functions are: len_pl, len_tl, len_cl, len_tol, len_pol 
    # For all of those: / sum_ipu_lgth
    # For len_cl: / len_pl and len_tl 
    # 'len_cl_t' => len_cl but duplicated for conv / part (naming issues)
    functions=['len_pl', 'len_tl', 'len_tol', 'len_pol', 'len_cl_p', 'len_cl_t', 'len_cl_over_p', 'len_cl_over_t']
    rmd.write(add_saver(functions))
    # Looping over functions
    for i,f in enumerate(functions):
        rmd.write("\n# " + f )
        #if plot_distrib:
        #    rmd.write(add_plot(f))
        sum_ipu_length = 'data_part' if re.search('t', f) else 'data_conv'
        rmd.write(add_description(f, sum_ipu_length))

    if excel_output is not None:
        rmd.write(print_saver(os.path.join(currdir,os.path.join('data_analysis/_exploration',excel_output))))
    # Close the file
    rmd.close()

if __name__ == '__main__':
    filename = 'explore_lilla.Rmd'
    basic_path = 'data/extracted_data_3.xlsx'
    align_path = 'data/extracted_align_data_2.xlsx'
    plot_distrib = True
    remove_subjects = [1,4,19,23]
    excel_output = 'lilla_exploration.xlsx'
    create_file(filename, basic_path, align_path, plot_distrib, remove_subjects, excel_output)