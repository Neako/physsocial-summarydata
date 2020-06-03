#!/usr/bin/env python
"""
R notebook creation automation for easier of analysis for new metrics

Plots R:
* https://philippmasur.de/2018/11/26/visualizing-interaction-effects/
* http://www.sthda.com/english/wiki/be-awesome-in-ggplot2-a-practical-guide-to-be-highly-effective-r-software-and-data-visualization

Example:
$ python data_analysis/generate_summary.Rmd.py lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth qt_discourse qt_feedback qt_filled_pause nratio_discourse nratio_feedback nratio_filled_pause mean_ipu_lgth speech_rate_min4 -l 'data/extracted_data_3.xlsx' -r 1 4 19 23
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
library(xlsx)
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

def add_data(linguistic_file=None, remove_subjects=[]):
    s = """
```{r}
# linguistic data
data <- read_excel('"""+linguistic_file+"""')
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[!(data$locutor %in% c("""+','.join([str(x) for x in remove_subjects])+""")),]
# data = data[which(data$locutor > 1),]
data$Trial2 = paste0('t', str_pad(data$conv_id_unif, 2, pad = "0"))
```
"""
    return s

def add_saver(functions):
    features = [f+'_'+state for f in functions for state in ['overall', 'part', 'conv']]
    s = """
```{r}
# extra columns will add themselves automatically - just creating structures
df_overall = data.frame(mean=numeric("""+str(len(features))+"""),
                std=numeric("""+str(len(features))+"""), 
                row.names = c('"""+"','".join(features)+"""'),
                stringsAsFactors=FALSE)
```
"""
    return s

def add_description(function_name):
    s = """
```{r}
# computing values
df_overall['"""+function_name+"""_overall', 'mean'] = mean(data$'"""+function_name+"""')
df_overall['"""+function_name+"""_overall', 'std'] = sd(data$'"""+function_name+"""')
s = summary(aov("""+function_name+"""~Agent*Trial2, data=data))[[1]]
for (a in c('Agent ', 'Trial2', 'Agent:Trial2')){
    for (b in c('F value', 'Pr(>F)')){
        c = ifelse(b == 'F value', paste0(trimws(a), '_z'), paste0(trimws(a), '_p'))
        df_overall['"""+function_name+"""_overall', c] = s[a,b]
    }
}
```\n
"""
    return s

def add_description_split(function_name):
    s = """
```{r}
temp1 = subset(data, select = c("locutor", "Trial2", "Agent", '"""+function_name+"""'), tier=='conversant')
temp2 = subset(data, select = c("locutor", "Trial2", "Agent", '"""+function_name+"""'), tier=='participant')
# adding resume
df_overall['"""+function_name+"""_conv', 'mean'] = mean(temp1$'"""+function_name+"""')
df_overall['"""+function_name+"""_conv', 'std'] = sd(temp1$'"""+function_name+"""')
df_overall['"""+function_name+"""_part', 'mean'] = mean(temp2$'"""+function_name+"""')
df_overall['"""+function_name+"""_part', 'std'] = sd(temp2$'"""+function_name+"""')

# creating merged data - ling
colnames(temp1) = c("locutor", "Trial2", "Agent", "data_conv")
colnames(temp2) = c("locutor", "Trial2", "Agent", "data_part")
merres = merge(temp1, temp2, by=c("locutor", "Trial2", "Agent"))
# computing values
s1 = summary(aov(data_conv~Agent*Trial2, data=merres))[[1]]
s2 = summary(aov(data_part~Agent*Trial2, data=merres))[[1]]
for (a in c('Agent ', 'Trial2', 'Agent:Trial2')){
    for (b in c('F value', 'Pr(>F)')){
        c = ifelse(b == 'F value', paste0(trimws(a), '_z'), paste0(trimws(a), '_p'))
        df_overall['"""+function_name+"""_part', c] = s2[a,b]
        df_overall['"""+function_name+"""_conv', c] = s1[a,b]
    }
}
```\n
"""
    return s

def add_plot(function_name):
    return """
```{r}
# plot
g <- ggplot(merres, aes(x = data_conv, y = data_part, color=Agent)) + 
        geom_point(alpha = 0.7) + 
        geom_density_2d(alpha=0.5) + 
        geom_smooth(method = "lm") +
        theme(legend.position="bottom") + xlim(0,max(merres$data_conv)) + ylim(0,max(merres$data_part)) +
        labs(x = "VI: """+function_name+""" conv",
            y = "VD: """+function_name+""" part",
            color = "Agent")
ggMarginal(g, type="densigram", margins = "both", groupColour = TRUE, fill="white")
```
"""

def print_saver(excel_output):
    return """
# Saver
```{r}
# Write the first data set in a new workbook
write.xlsx(df_overall, file = '"""+excel_output+"""',
      sheetName = "summary", append = FALSE)
```
"""

def create_file(functions, filename, ling_path, remove_subjects, excel_output):
    # read path to create file
    currdir = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join('data_analysis/_exploration', filename), 'w')

    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    ling_path = None if ling_path is None else os.path.join(currdir,ling_path)
    rmd.write(add_data(ling_path, remove_subjects))
    rmd.write(add_saver(functions))
    for f in functions:
        rmd.write("\n# " + f )
        rmd.write(add_description(f))
        rmd.write(add_description_split(f))
        rmd.write(add_plot(f))
    rmd.write(print_saver(os.path.join(currdir,os.path.join('data_analysis/_exploration',excel_output))))
    # Close the file
    rmd.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-o', type=str, default='generated_summary.Rmd')
    parser.add_argument('--excel_output', '-e', type=str, default='summary.xlsx')
    # lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth qt_discourse qt_feedback qt_filled_pause nratio_discourse nratio_feedback nratio_filled_pause mean_ipu_lgth speech_rate_min4
    parser.add_argument('functions', nargs='+', type=str)
    parser.add_argument('--ling_path', '-l', type=str, default=None)
    parser.add_argument('--remove_subjects', '-r', type=int, nargs='+', default=[])
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.file_name, args.ling_path, args.remove_subjects, args.excel_output)