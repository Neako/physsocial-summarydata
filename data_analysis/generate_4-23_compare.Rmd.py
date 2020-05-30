"""
R notebook creation automation - comparison of participants to whole
Automatisation necessary for ggplots graph are not printed in loops

Example:
$ python data_analysis/generate_4-23_compare.Rmd.py 4 23 -f lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth ratio_discourse ratio_feedback ratio_filled_pause mean_ipu_lgth -l "data/extracted_data_3.xlsx" -p True
"""

import sys
import os
import argparse
import ast

def add_header():
    return """---
title: "R Notebook"
output:
html_document:
    df_print: paged
---

"""

def add_libraries(additional_libaries = []):
    s = """```{r}
library(readxl)
library(sjPlot)
library(ggplot2)
library(lme4)
library(stringr)
library(ggExtra)
library(dplyr)
"""
    for lib in additional_libaries:
        s += "library(" + lib + ")\n"
    s +="""```\n"""
    return s

def add_data(linguistic_file, compare_subjects, features=None):
    s = """
```{r}
# linguistic data
data <- read_excel('"""+linguistic_file+"""')
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[which(data$locutor > 1),]
data$Trial2 = paste0('t', str_pad(data$conv_id_unif, 2, pad = "0"))

data_wo = data[!(data$locutor %in% c("""+','.join([str(x) for x in compare_subjects])+""")),]
data_w = data[(data$locutor %in% c("""+','.join([str(x) for x in compare_subjects])+""")),]
# adding column to main data to know which is which
data$is_compared = ifelse((data$locutor %in% c("""+','.join([str(x) for x in compare_subjects])+""")), 1, 0)
"""
    if features is not None:
        s += """
df = data.frame(test_full=numeric("""+str(len(features))+"""),
                test_without=numeric("""+str(len(features))+"""), 
                row.names = c('"""+"','".join(features)+"""'),
                stringsAsFactors=FALSE)
```
"""
    else:
        s += """\n```\n"""
    return s

def add_plot(function_name):
    return """```{r error=TRUE}
ggplot(data[which(data$tier == 'participant'),], aes(x = """+function_name+""", color=factor(is_compared))) + 
    geom_histogram(aes(y=..density..), alpha=0.5, fill="white") + 
    geom_density(alpha=.2)
```
"""

def add_wilcox(function_name, add_summary):
    s = """```{r error=TRUE}
tmp_w = subset(data_w, select = c("locutor", "conv_id_unif", "Agent", '"""+function_name+"""'), tier=='participant')
tmp_wo = subset(data_wo, select = c("locutor", "conv_id_unif", "Agent", '"""+function_name+"""'), tier=='participant')
tmp_full = subset(data, select = c("locutor", "conv_id_unif", "Agent", '"""+function_name+"""'), tier=='participant')
w1 = wilcox.test(pull(tmp_w, '"""+function_name+"""'), pull(tmp_wo, '"""+function_name+"""'))
w2 = wilcox.test(pull(tmp_w, '"""+function_name+"""'), pull(tmp_full, '"""+function_name+"""'))
print(w1)
print(w2)
"""
    if add_summary:
        s += """
df['"""+function_name+"""', 'test_full'] = w2$p.value
df['"""+function_name+"""', 'test_without'] = w1$p.value
```
"""
    else:
        s += """\n```\n"""
    return s

def add_table():
    return """

# Summary
```{r}
df
```
"""

def create_file(functions, filename, ling_path, plot_distrib, add_summary, compare_subjects):
    # read path to create file
    currdir = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join('data_analysis/_exploration', filename), 'w')
    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    ling_path = None if ling_path is None else os.path.join(currdir,ling_path)
    rmd.write(add_data(ling_path, compare_subjects, features=(functions if add_summary is not None else None)))
    for f in functions:
        rmd.write("\n# {} \n".format(f))
        if plot_distrib:
            rmd.write(add_plot(f))
        rmd.write(add_wilcox(f, add_summary))
    if add_summary:
        rmd.write(add_table())

    # Close the file
    rmd.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-o', type=str, default='comparison_4-23.Rmd')
    parser.add_argument('compare_subjects', type=int, nargs='+')
    parser.add_argument('--functions', '-f', nargs='+', type=str, default=['sum_ipu_lgth'])
    parser.add_argument('--ling_path', '-l', type=str, default=None)
    parser.add_argument('--plot_distrib', '-p', type=bool, default=False)
    parser.add_argument('--add_summary', '-a', type=bool, default=True)
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.file_name, args.ling_path, args.plot_distrib, args.add_summary, args.compare_subjects)