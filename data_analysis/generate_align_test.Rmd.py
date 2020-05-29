"""
R notebook creation automation for easier analysis of new alignment metrics

Plots R:
* https://philippmasur.de/2018/11/26/visualizing-interaction-effects/
* http://www.sthda.com/english/wiki/be-awesome-in-ggplot2-a-practical-guide-to-be-highly-effective-r-software-and-data-visualization

Example:
$ python data_analysis/generate_align_test.Rmd.py lilla log_lilla -l "data/extracted_align_data.xlsx" -p True
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
data = data[!(data$locutor %in% c("""+','.join(remove_subjects)+""")),]
# data = data[which(data$locutor > 1),]
data$Trial2 = paste0('t', str_pad(data$conv_id_unif, 2, pad = "0"))
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
        labs(x = "VI: """+function_name+""" Conv",
            y = "VD: """+function_name+""" Part",
            color = "Agent")
ggMarginal(g, type="density", margins = "both", groupColour = TRUE)

# change names to avoid later confusion
names(data_convprime)[names(data_convprime) == 'data_conv'] = '"""+function_name+"""' 
names(data_partprime)[names(data_partprime) == 'data_part'] = '"""+function_name+"""'
```
"""
    return s

def add_models(function_name, prime, formula_ling, formula_neuro, has_neuro=False):
    s = """
```{r error=TRUE}
# applying mixed model
mdl = lmer('"""+formula_ling.replace('function', function_name)+"""', data = data_"""+prime+"""prime)
print(summary(mdl))
tab_model(mdl, title = '"""+function_name+"""')
#print(confint(mdl))

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

def create_file(functions, primes, filename, neuro_path, ling_path, plot_distrib, formula_ling, formula_neuro, remove_subjects):
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
    for f in functions:
        for prime in primes:
            rmd.write("\n# {} {}prime\n".format(f, prime[:4]))
            if plot_distrib:
                rmd.write(add_plot(f, prime[:4]))
            rmd.write(add_models(f, prime[:4], formula_ling, formula_neuro, has_neuro = (neuro_path is not None)))
        if plot_distrib:
            rmd.write(add_mixedplot(f))

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
    args = parser.parse_args()
    print(args)
    create_file(args.functions, args.primes, args.file_name, args.neuro_path, args.ling_path, args.plot_distrib, args.formula_ling, args.formula_neuro, args.remove_subjects)