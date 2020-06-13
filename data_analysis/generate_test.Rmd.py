#!/usr/bin/env python
"""
R notebook creation automation for easier of analysis for new metrics

Plots R:
* https://philippmasur.de/2018/11/26/visualizing-interaction-effects/
* http://www.sthda.com/english/wiki/be-awesome-in-ggplot2-a-practical-guide-to-be-highly-effective-r-software-and-data-visualization

Example:
$ python data_analysis/generate_test.Rmd.py speech_rate speech_rate_mean speech_rate_min speech_rate_max -o "speech_rate.Rmd" -l "data/extracted_data.xlsx" -p True -fl "data_part ~ Agent*data_conv*Trial +(1|locutor)"
$ python data_analysis/generate_test.Rmd.py lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth ratio_discourse ratio_feedback ratio_filled_pause mean_ipu_lgth -o "basic_features.Rmd" -l "data/extracted_data.xlsx" -p True -n "data_neuro/Broca.txt"
$ python data_analysis/generate_test.Rmd.py lexical_richness linguistic_complexity ratio_silence_lgth sum_ipu_lgth nratio_discourse nratio_feedback nratio_filled_pause mean_ipu_lgth -o "basic_features_srm_2.Rmd" -l "data/extracted_data_3.xlsx" -p True -r 1 4 19 23 -ee True -eo "summary.xlsx"
"""

import sys
import os
import argparse
import ast
# local functions
from generate_utils import *

def add_mergedata(function_name, formula_ling, formula_neuro, has_neuro=False, save_data=False, is_first=False):
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

"""
    if save_data:
        s += """# saving data
s = summary(mdl)[['coefficients']]
s = data.frame(s)
s$Feature = '"""+function_name+"""'
l = data.frame(suppressWarnings(confint(mdl)))[5:9,]
"""
        if is_first:
            s += """df_overall = cbind(s,l)
"""
        else:
            s += """df_overall = rbind(df_overall, cbind(s,l))
"""
    s += """
# saving other features
data_r = merres[which(merres$Agent == "R"),]
data_h = merres[which(merres$Agent == "H"),]
for (pc in c('part', 'conv')){
    df2[paste0('"""+function_name+"""_', pc), 'mean'] = mean(merres[[paste0('data_',pc)]])
    df2[paste0('"""+function_name+"""_', pc), 'std'] = sd(merres[[paste0('data_',pc)]])
    df2[paste0('"""+function_name+"""_', pc), 'mean_r'] = mean(data_r[[paste0('data_',pc)]])
    df2[paste0('"""+function_name+"""_', pc), 'std_r'] = sd(data_r[[paste0('data_',pc)]])
    df2[paste0('"""+function_name+"""_', pc), 'mean_h'] = mean(data_h[[paste0('data_',pc)]])
    df2[paste0('"""+function_name+"""_', pc), 'std_h'] = sd(data_h[[paste0('data_',pc)]])
}
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

def add_saver(functions):
    features = [f+'_'+state for f in functions for state in ['part', 'conv']]
    s = """
```{r}
# extra columns will add themselves automatically - just creating structures
df2 = data.frame(mean=numeric("""+str(len(features))+"""),
                std=numeric("""+str(len(features))+"""), 
                mean_r=numeric("""+str(len(features))+"""),
                std_r=numeric("""+str(len(features))+"""),
                mean_h=numeric("""+str(len(features))+"""),
                std_h=numeric("""+str(len(features))+"""),
                row.names = c('"""+"','".join(features)+"""'),
                stringsAsFactors=FALSE)
```
"""
    return s

def create_file(functions, filename, neuro_path, ling_path, plot_distrib, formula_ling, formula_neuro, remove_subjects, excel_output, excel_exists):
    # read path to create file
    currdir = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join('data_analysis/_exploration', filename), 'w')

    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    #neuro_path = None if neuro_path is None else os.path.join(currdir,neuro_path)
    #ling_path = None if ling_path is None else os.path.join(currdir,ling_path)
    rmd.write(add_data(neuro_path, ling_path, remove_subjects))
    rmd.write(add_saver(functions))

    for i,f in enumerate(functions):
        rmd.write("\n# " + f )
        if plot_distrib:
            rmd.write(add_description(f))
        rmd.write(add_mergedata(f, formula_ling, formula_neuro, has_neuro = (neuro_path is not None), save_data=(excel_output is not None), is_first=(i==0)))
        rmd.write(add_plot(f))

    if excel_output is not None:
        # excel_output = os.path.join(currdir,os.path.join('data_analysis/_exploration',excel_output))
        dfs = {"df_overall":"models", "df2":"hr_comparison"}
        rmd.write(print_saver(excel_output, dfs, excel_exists))
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
    parser.add_argument('--remove_subjects', '-r', type=int, nargs='+', default=[])
    parser.add_argument('--formula_ling', '-fl', type=str, default="data_part ~ data_conv * Agent + Trial + (1 + Trial | locutor)")
    parser.add_argument('--formula_neuro', '-fn', type=str, default="bold ~ data_part * Agent + Trial + (1 + Trial | locutor)")
    parser.add_argument('--excel_output', '-eo', type=str, default=None)
    parser.add_argument('--excel_exists', '-ee', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.file_name, args.neuro_path, args.ling_path, args.plot_distrib, args.formula_ling, args.formula_neuro, args.remove_subjects, args.excel_output, args.excel_exists)