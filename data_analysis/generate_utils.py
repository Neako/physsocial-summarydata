#!/usr/bin/env python
"""
Storing common functions in generate_XXX.Rmd.py to avoid code duplication
"""
import sys
import os
import argparse
import ast

###### Header and library loading
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

###### Loading data
def add_data(neuro_file=None, linguistic_file=None, remove_subjects=[], is_align=False):
    s = """
```{r}
# linguistic data
if (rstudioapi::isAvailable()){
  currdir = dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  currdir = getwd()
}
file_path = file.path(dirname(dirname(currdir)), '"""+linguistic_file+"""')
data <- read_excel(file_path)
# Handling data
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[!(data$locutor %in% c("""+','.join([str(x) for x in remove_subjects])+""")),]
# Adding / renaming columns
data$Trial = data[[ifelse("Trial" %in% colnames(data), "Trial", "conv_id_unif")]] # rename if not in it
data$Trial2 = paste0('t', str_pad(data$Trial, 2, pad = "0"))
"""
    if is_align:
        s += """# Creating dfs
data_convprime = data[which(data$prime == "conversant"),]
data_partprime = data[which(data$prime == "participant"),]
"""
    if neuro_file is not None:
        s += """# neuro data
file_path = file.path(dirname(dirname(dirname(rstudioapi::getSourceEditorContext()$path))), '"""+neuro_file+"""')
broca = read.table(file = file_path, sep = '\\t', header = FALSE)
colnames(broca) = c("area", "locutor", "session", "image", "bold", "Agent", "Trial")
broca = broca[which(broca$Agent != ""),] # remove last line: count
broca$Agent = ifelse(as.numeric(broca$Agent) == 2,"H","R") # "Factor w/ 3 levels" bc of last line
broca$Trial = broca$Trial-1
```
"""
    else:
        s += "```\n"
    return s

###### Adding plots
def add_mixedplot(function_name, is_align=False):
    s1 = """
```{r error=TRUE}
"""
    s_a1 = """names(data_convprime)[names(data_convprime) == '"""+function_name+"""'] = 'data_conv'
names(data_partprime)[names(data_partprime) == '"""+function_name+"""'] = 'data_part'
merres = merge(data_convprime[,c('locutor', "Trial", "Agent", 'data_conv')], data_partprime[,c('locutor', "Trial", "Agent", 'data_part')], by=c("locutor", "Trial", "Agent"))
"""
    s2 = """
# plot
g <- ggplot(merres, aes(x = data_conv, y = data_part, color=Agent)) + 
        geom_point(alpha = 0.7) + 
        geom_density_2d(alpha=0.5) {}+ 
        theme(legend.position="bottom") + xlim(0,max(merres$data_conv)) + ylim(0,max(merres$data_part)) +
        labs(x = "VI: """+function_name+""" conv",
            y = "VD: """+function_name+""" part",
            color = "Agent")
ggMarginal(g, type="densigram", margins = "both", groupColour = TRUE, fill="white")
"""
    s_a2 = """
# change names to avoid later confusion
names(data_convprime)[names(data_convprime) == 'data_conv'] = '"""+function_name+"""' 
names(data_partprime)[names(data_partprime) == 'data_part'] = '"""+function_name+"""'
"""
    s3 = """```
"""
    if is_align:
        return s1 + s_a1 + s2.format('') + s_a2 + s3
    else:
        return s1 + s2.format("+ geom_smooth(method = 'lm') ") + s3

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

def add_description(function_name, is_align=False):
    fg = "prime" if is_align else "tier"
    s = """
```{r error=TRUE}
ggplot(data, aes(x = """+function_name+""", color=Agent)) + facet_grid("""+fg+""" ~ .) + geom_histogram(aes(y=..density..), alpha=0.5, fill="white") + geom_density(alpha=.2)
# Trial
ggplot(data, aes(x = Trial2, y = """+function_name+""", color=Agent)) + facet_grid("""+fg+""" ~ .) + geom_boxplot()
ggplot(data, aes(x = Trial, y = """+function_name+""", color=Agent)) + geom_point() + geom_smooth(method="lm") + facet_grid("""+fg+""" ~ .)
# means
ggplot(data, 
       aes(x = Agent,
           fill = Agent,  
           y = """+function_name+""")) +
  stat_summary(fun.y = mean,
               geom = "bar") +
  stat_summary(fun.ymin = function(x) mean(x) - sd(x), 
               fun.ymax = function(x) mean(x) + sd(x), 
               geom="errorbar", 
               width = 0.25) +
  facet_wrap(~"""+fg+""") +
  labs(x = "Agent",
       y = '"""+function_name+"""')
```\n
"""
    return s

###### Save data
def print_saver(excel_output, dfs, excel_exists=False):
    """Saves dfs to excel

    Input:
    -------
    excel_output: str
        excel path
    dfs: dict
        aforementioned dfs, shape {df_name:sheet_name}
    excel_exists: bool

    Output:
    -------
    s: str
    """
    key_first = list(dfs.keys())[0]
    s = """
# Saver
```{r error=TRUE}
if (rstudioapi::isAvailable()){
  file_path = file.path(dirname(rstudioapi::getSourceEditorContext()$path), '"""+excel_output+"""')
} else {
  file_path = file.path(getwd(), '"""+excel_output+"""')
}
# Write the first data set in a new workbook
write.xlsx("""+key_first+""", file = file_path,
      sheetName = '"""+dfs.pop(key_first)+"""', append = """+str(excel_exists).upper()+""")
# Write others sheets
"""
    for k,v in dfs.items():
        s += """
write.xlsx("""+k+""", file = file_path,
      sheetName = '"""+v+"""', append = TRUE)
"""
    s += """
```
"""
    return s