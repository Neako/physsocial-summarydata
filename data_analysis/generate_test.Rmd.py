#!/usr/bin/env python
"""
R notebook creation automation for easier of analysis for new metrics

Plots R:
* https://philippmasur.de/2018/11/26/visualizing-interaction-effects/
* http://www.sthda.com/english/wiki/be-awesome-in-ggplot2-a-practical-guide-to-be-highly-effective-r-software-and-data-visualization

Example:
$ python data_analysis/generate_test.Rmd.py speech_rate speech_rate_mean speech_rate_min speech_rate_max -o "speech_rate.Rmd" -l "data/extracted_data.xlsx" -p True -fl "data_part ~ Agent*data_conv*Trial +(1|locutor)"

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
"""
    for lib in additional_libaries:
        s += "library(" + lib + ")\n"
    s +="""```\n"""
    return s

def add_data(neuro_file=None, linguistic_file=None):
    s = """
```{r}
# linguistic data
data <- read_excel('"""+linguistic_file+"""')
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[which(data$locutor > 1),]
data$Trial2 = paste0('t', str_pad(data$conv_id_unif, 2, pad = "0"))
"""
    if neuro_file is not None:
        s += """# neuro data
broca = read.table(file = '"""+neuro_file+"""', sep = '\\t', header = FALSE)
colnames(broca) = c("area", "locutor", "session", "image", "bold", "Agent", "Trial")
broca = broca[which(broca$Agent != ""),] # remove last line: count
broca$Agent = ifelse(as.numeric(broca$Agent) == 2,"H","R") # "Factor w/ 3 levels" bc of last line
broca$Trial = broca$Trial-1
```
"""
    else:
        s += "```\n"
    return s

def add_mergedata(function_name, formula_ling, formula_neuro, has_neuro=False):
    s = """
```{r}
# creating merged data - ling
temp1 = subset(data, select = c("locutor", "conv_id_unif", "Agent", '"""+function_name+"""'), tier=='conversant')
colnames(temp1) = c("locutor", "Trial", "Agent", "data_conv")
temp2 = subset(data, select = c("locutor", "conv_id_unif", "Agent", '"""+function_name+"""'), tier=='participant')
colnames(temp2) = c("locutor", "Trial", "Agent", "data_part")
merres = merge(temp1, temp2, by=c("locutor", "Trial", "Agent"))
# applying mixed model
mdl = lmer('"""+formula_ling+"""', data = merres)
print(summary(mdl))
tab_model(mdl, title = paste("part ~ conv ", '"""+function_name+"""'))
#print(confint(mdl))

"""
    if has_neuro:
        s += """# creating merged data - neuro
merneuro = merge(merres, broca, by=c("locutor", "Trial", "Agent"))
model = lmer('"""+formula_neuro+"""', data = merneuro)
print(summary(model))
tab_model(model, title = paste("bold ~ ", '"""+function_name+"""'))
```

"""
    else:
        s += "```\n"
    return s

def add_plot(function_name):
    return """```{r}
# Setting up the building blocks
basic_plot <- ggplot(merres,
       aes(x = data_conv,
           y = data_part,
           color = Agent)) +
  theme_bw()

# Colored scatterplot and regression lines
basic_plot +
  geom_point(alpha = .3, 
             size = .9) +
  geom_smooth(method = "lm") +
  labs(x = "VI: """+function_name+""" Conv",
       y = "VD: """+function_name+""" Part",
       color = "Agent")
```

```{r}
# second plot
g <- ggplot(merres, aes(x = data_conv, y = data_part, color=Agent)) + 
        geom_point(alpha = 0.7) + 
        geom_density_2d(alpha=0.5) + 
        theme(legend.position="bottom") + xlim(0,max(merres$data_conv)) + ylim(0,max(merres$data_part)) +
        labs(x = "VI: """+function_name+""" Conv",
            y = "VD: """+function_name+""" Part",
            color = "Agent")
ggMarginal(g, type="density", margins = "both", groupColour = TRUE)
```
"""

def add_description(function_name):
    s = """
```{r}
ggplot(data, aes(x = """+function_name+""", color=Agent)) + facet_grid(tier ~ .) + geom_histogram(aes(y=..density..), alpha=0.5, fill="white") + geom_density(alpha=.2)
ggplot(data, aes(x = Trial2, y = """+function_name+""", color=Agent)) + facet_grid(tier ~ .) + geom_boxplot()
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
  facet_wrap(~tier) +
  labs(x = "Agent",
       y = '"""+function_name+"""')
```\n
"""
    return s

def create_file(functions, filename, neuro_path, ling_path, plot_distrib, formula_ling, formula_neuro):
    # read path to create file
    currdir = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join('data_analysis/_exploration', filename), 'w')

    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    neuro_path = None if neuro_path is None else os.path.join(currdir,neuro_path)
    ling_path = None if ling_path is None else os.path.join(currdir,ling_path)
    rmd.write(add_data(neuro_path, ling_path))
    for f in functions:
        rmd.write("\n# " + f )
        if plot_distrib:
            rmd.write(add_description(f))
        rmd.write(add_mergedata(f, formula_ling, formula_neuro, has_neuro = (neuro_path is not None)))
        rmd.write(add_plot(f))

    # Close the file
    rmd.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-o', type=str, default='generated_stats.Rmd')
    # lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth ratio_discourse ratio_feedback ratio_filled_pause mean_ipu_lgth
    # speech_rate speech_rate_mean speech_rate_min speech_rate_max
    parser.add_argument('functions', nargs='+', type=str)
    parser.add_argument('--ling_path', '-l', type=str, default=None)
    parser.add_argument('--neuro_path', '-n', type=str, default=None)
    parser.add_argument('--plot_distrib', '-p', type=bool, default=False)
    parser.add_argument('--formula_ling', '-fl', type=str, default="data_part ~ data_conv * Agent + Trial + (1 + Trial | locutor)")
    parser.add_argument('--formula_neuro', '-fn', type=str, default="bold ~ data_part * Agent + Trial + (1 + Trial | locutor)")
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.file_name, args.neuro_path, args.ling_path, args.plot_distrib, args.formula_ling, args.formula_neuro)