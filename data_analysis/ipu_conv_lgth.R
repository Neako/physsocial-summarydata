library(readxl)
library(ggplot2)
library(wesanderson)
file_path = file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)), "data/ipu_conv.xlsx")
ipu_conv <- read_excel(file_path)

# plots for idea
ggplot(ipu_conv[which((ipu_conv$label == "c'est une campagne pour les pesticides") 
                      & (ipu_conv$agent == "R")),], 
       aes(x = duration)) + geom_histogram(aes(y=..density..), alpha=0.5) + 
  geom_density(alpha=.2)

ggplot(ipu_conv[which((ipu_conv$label == "ah oui") & (ipu_conv$agent == "R")),], 
       aes(x = duration)) + geom_histogram(aes(y=..density..), alpha=0.5) + 
  geom_density(alpha=.2)

ggplot(ipu_conv[which((ipu_conv$label == "salut") & (ipu_conv$agent == "R")),], 
       aes(x = duration)) + geom_histogram(aes(y=..density..), alpha=0.5) + 
  geom_density(alpha=.2)

# plotting whole data
ipu_conv_r <- ipu_conv[which((ipu_conv$agent == "R")),]
means = aggregate(ipu_conv_r$duration, FUN=mean, 
                  by=list(label=ipu_conv_r$label)) #getting the mean for each experimental cell
means$min = as.numeric(aggregate(ipu_conv_r$duration, FUN=min, 
                                by=list(label=ipu_conv_r$label))$x)
means$max = as.numeric(aggregate(ipu_conv_r$duration, FUN=max, 
                                 by=list(label=ipu_conv_r$label))$x)
means$sd = as.numeric(aggregate(ipu_conv_r$duration, FUN=sd, 
                                 by=list(label=ipu_conv_r$label))$x)
means$count = as.numeric(aggregate(ipu_conv_r$duration, FUN=length, 
                                by=list(label=ipu_conv_r$label))$x)
means$diff = means$max - means$min
means$nchar = nchar(means$label)

head(means)

# Setting up the building blocks
ggplot(means, aes(x = count,
                  y = diff, color=nchar)) +
  theme_bw() +
  geom_point(alpha = .8, 
             size = .9) +
  geom_smooth(method = "lm") +
  labs(x = "Number of occurences (log scale)",
       y = "min-max difference") + 
  scale_x_continuous(trans='log2')

# glm pour les stats
mdl = lm('diff ~ log(count) + nchar', data = means)
print(summary(mdl))
tab_model(mdl)

# rassembler moyenne et diff à la moyenne pour chaque segment
ipu_conv_r$diff_to_mean = numeric(length(ipu_conv_r$duration))
for (i in seq(1,length(ipu_conv_r$label))){
  ipu_conv_r$diff_to_mean[i] = abs(ipu_conv_r$duration[i] - 
                                     means[which(means$label == ipu_conv_r$label[i]),]$x)
}
ipu_conv_r$file_name = paste0('S', str_pad(ipu_conv_r$locutor, 2, pad = "0"),
                              '_Sess', (ipu_conv_r$trial-1)%%4+1,
                              '_CONV2_', str_pad(((ipu_conv_r$trial-1)%%3+1)*2, 3, pad = "0"),
                              '-conversant.TextGrid')
# S09_Sess3_CONV2_006-conversant.TextGrid
ipu_conv_r[which(ipu_conv_r$diff_to_mean > 0.5),c('label', 'diff_to_mean', 'file_name')]

# debit = f(duration)
file_path = file.path(dirname(dirname(rstudioapi::getSourceEditorContext()$path)), "data/ipu_all.xlsx")
ipu_all <- read_excel(file_path)
ipu_all_conv = ipu_all[which(ipu_all$tier == 'conversant'),]
ipu_all_part = ipu_all[which(ipu_all$tier == 'participant'),]
ipu_all_part$nb_vowels = ipu_all_part$speech_rate * ipu_all_part$duration
ggplot(ipu_all_part, aes(x = duration,
                         y = speech_rate, color=agent)) +
  scale_color_gradientn(colours = rainbow(5)) +
  geom_point(alpha = .8, 
             size = .9) +
  geom_smooth(method = "lm") +
  labs(x = "IPU duration",
       y = "Speech rate") 
ggplot(ipu_all_part, aes(x = duration,
                  y = speech_rate, color=nb_vowels)) +
  scale_color_gradientn(colours = wes_palette(n=5, name="Zissou1"), trans = "pseudo_log") +
  geom_point(alpha = .8, 
             size = .9) +
  labs(x = "IPU duration",
       y = "Speech rate",
       title = "Speech rate variation by number of vowels in text") +
  xlim(0,9)
# trans: "asn", "atanh", "boxcox", "date", "exp", "hms", "identity", "log", "log10", "log1p", "log2", "logit", 
# "modulus", "probability", "probit", "pseudo_log", "reciprocal", "reverse", "sqrt" and "time"
