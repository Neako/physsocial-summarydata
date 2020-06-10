"""
R notebook creation automation for easier analysis of new alignment metrics

Plots R:
* https://philippmasur.de/2018/11/26/visualizing-interaction-effects/
* http://www.sthda.com/english/wiki/be-awesome-in-ggplot2-a-practical-guide-to-be-highly-effective-r-software-and-data-visualization

Example:
$ python data_analysis/generate_align_test.Rmd.py lilla log_lilla -l "data/extracted_align_data.xlsx" -p True
$ python data_analysis/generate_align_test.Rmd.py lilla -l "data/extracted_align_data.xlsx" -p True -r 1 4 19 23 -sm "summary_align.xlsx"
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

def add_data(neuro_file=None, linguistic_file=None, remove_subjects=[]):
    s = """
```{r}
# linguistic data
data <- read_excel('"""+linguistic_file+"""')
data$Agent = ifelse(data$conv == 1,"H","R")
data = data[!(data$locutor %in% c("""+','.join([str(x) for x in remove_subjects])+""")),]
# data = data[which(data$locutor > 1),]
data$Trial2 = paste0('t', str_pad(data$Trial, 2, pad = "0"))
data_convprime = data[which(data$prime == "conversant"),]
data_partprime = data[which(data$prime == "participant"),]
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

def add_plot(function_name, prime):
    return """```{r error=TRUE}
ggplot(data_"""+prime+"""prime, aes(x = """+function_name+""", color=Agent)) + geom_histogram(aes(y=..density..), alpha=0.5, fill="white") + geom_density(alpha=.2)
ggplot(data_"""+prime+"""prime, aes(x = Trial2, y = """+function_name+""", color=Agent)) + geom_boxplot()
ggplot(data_"""+prime+"""prime, 
       aes(x = Agent,
           fill = Agent,  
           y = """+function_name+""")) +
  stat_summary(fun.y = mean,
               geom = "bar") +
  stat_summary(fun.ymin = function(x) mean(x) - sd(x), 
               fun.ymax = function(x) mean(x) + sd(x), 
               geom="errorbar", 
               width = 0.25) +
  labs(x = "Agent",
       y = '"""+function_name+"""')
```
"""

def add_mixedplot(function_name):
    s = """
```{r error=TRUE}
names(data_convprime)[names(data_convprime) == '"""+function_name+"""'] = 'data_conv'
names(data_partprime)[names(data_partprime) == '"""+function_name+"""'] = 'data_part'
merres = merge(data_convprime[,c('locutor', "Trial", "Agent", 'data_conv')], data_partprime[,c('locutor', "Trial", "Agent", 'data_part')], by=c("locutor", "Trial", "Agent"))
# plot
g <- ggplot(merres, aes(x = data_conv, y = data_part, color=Agent)) + 
        geom_point(alpha = 0.7) + 
        geom_density_2d(alpha=0.5) + 
        theme(legend.position="bottom") + xlim(0,max(merres$data_conv)) + ylim(0,max(merres$data_part)) +
        labs(x = "VI: """+function_name+""" conv",
            y = "VD: """+function_name+""" part",
            color = "Agent")
ggMarginal(g, type="densigram", margins = "both", groupColour = TRUE)

# change names to avoid later confusion
names(data_convprime)[names(data_convprime) == 'data_conv'] = '"""+function_name+"""' 
names(data_partprime)[names(data_partprime) == 'data_part'] = '"""+function_name+"""'
```
"""
    return s

def add_models(function_name, prime, formula_ling, formula_neuro, has_neuro=False, save_data=False, is_first=False):
    s = """
```{r error=TRUE}
# applying mixed model
mdl = lmer('"""+formula_ling.replace('function', function_name)+"""', data = data_"""+prime+"""prime)
print(summary(mdl))
tab_model(mdl, title = '"""+function_name+"""')
#print(confint(mdl))

"""
    if save_data:
        s += """# saving data
s = summary(mdl)[['coefficients']]
s = data.frame(s)
s$Feature = '"""+function_name+'_'+prime+"""'
l = data.frame(confint(mdl))[3:6,]
"""
        if is_first:
            s += """df_model = cbind(s,l)
"""
        else:
            s += """df_model = rbind(df_model, cbind(s,l))
"""
        s += """
# saving other features
data_r = data_"""+prime+"""prime[which(data_"""+prime+"""prime$Agent == "R"),]
data_h = data_"""+prime+"""prime[which(data_"""+prime+"""prime$Agent == "H"),]
df_overall['"""+function_name+"""_"""+prime+"""', 'mean'] = mean(data$'"""+function_name+"""')
df_overall['"""+function_name+"""_"""+prime+"""', 'std'] = sd(data$'"""+function_name+"""')
df_overall['"""+function_name+"""_"""+prime+"""', 'mean_r'] = mean(data_r$'"""+function_name+"""')
df_overall['"""+function_name+"""_"""+prime+"""', 'std_r'] = sd(data_r$'"""+function_name+"""')
df_overall['"""+function_name+"""_"""+prime+"""', 'mean_h'] = mean(data_h$'"""+function_name+"""')
df_overall['"""+function_name+"""_"""+prime+"""', 'std_h'] = sd(data_h$'"""+function_name+"""')
"""
    if has_neuro:
        s += """# creating merged data - neuro
merneuro = merge(data_"""+prime+"""prime, broca, by=c("locutor", "Trial", "Agent"))
model = lmer('"""+formula_ling.replace('function', function_name)+"""', data = merneuro)
print(summary(model))
tab_model(model, title = paste("bold ~ ", '"""+function_name+"""'))
```

"""
    else:
        s += "```\n"
    return s

def add_saver(functions, primes):
    features = [f+'_'+state[:4] for f in functions for state in primes]
    s = """
```{r}
# extra columns will add themselves automatically - just creating structures
df_overall = data.frame(mean=numeric("""+str(len(features))+"""),
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

def print_saver(excel_output):
    return """
# Saver
```{r error=TRUE}
# Write the first data set in a new workbook
write.xlsx(df_overall, file = '"""+excel_output+"""',
      sheetName = "summary", append = FALSE)
# Write model to last sheet:
write.xlsx(df_model, file = '"""+excel_output+"""',
      sheetName = "model", append = TRUE)
```
"""

def create_file(functions, primes, filename, neuro_path, ling_path, plot_distrib, formula_ling, formula_neuro, remove_subjects, add_summary):
    # read path to create file
    currdir = os.path.dirname(os.path.realpath(__file__)).replace('/data_analysis','')
    # Open the file with writing permission
    rmd = open(os.path.join('data_analysis/_exploration', filename), 'w')

    # Write data to the file
    rmd.write(add_header())
    rmd.write(add_libraries())
    neuro_path = None if neuro_path is None else os.path.join(currdir,neuro_path)
    ling_path = None if ling_path is None else os.path.join(currdir,ling_path)
    rmd.write(add_data(neuro_path, ling_path, remove_subjects))
    if add_summary:
        rmd.write(add_saver(functions, primes))
    for i, f in enumerate(functions):
        for j, prime in enumerate(primes):
            rmd.write("\n# {} {}prime\n".format(f, prime[:4]))
            if plot_distrib:
                rmd.write(add_plot(f, prime[:4]))
            rmd.write(add_models(f, prime[:4], formula_ling, formula_neuro, has_neuro = (neuro_path is not None), save_data=(add_summary is not None), is_first=((i+j)==0))) # could be i+j, i+2*j, whichever, the only 0 is (0,0)
        if plot_distrib:
            rmd.write(add_mixedplot(f))

    if add_summary:
        rmd.write(print_saver(os.path.join(currdir,os.path.join('data_analysis/_exploration',add_summary))))
    # Close the file
    rmd.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', '-o', type=str, default='generated_align_stats.Rmd')
    # lilla log_lilla
    parser.add_argument('functions', nargs='+', type=str)
    parser.add_argument('--primes', '-c', nargs='+', type=str, default=['conversant', 'participant'])
    parser.add_argument('--ling_path', '-l', type=str, default=None)
    parser.add_argument('--neuro_path', '-n', type=str, default=None)
    parser.add_argument('--plot_distrib', '-p', type=bool, default=False)
    parser.add_argument('--remove_subjects', '-r', type=int, nargs='+', default=[])
    parser.add_argument('--formula_ling', '-fl', type=str, default="function ~ Agent * Trial + (1 | locutor)")
    parser.add_argument('--formula_neuro', '-fn', type=str, default="bold ~ function * Agent + Trial + (1 + Trial | locutor)")
    parser.add_argument('--add_summary', '-sm', type=str, default=None)
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.primes, args.file_name, args.neuro_path, args.ling_path, args.plot_distrib, args.formula_ling, args.formula_neuro, args.remove_subjects, args.add_summary)