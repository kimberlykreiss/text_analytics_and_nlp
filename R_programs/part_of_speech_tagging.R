#####################################################
#
# This program uses the custom tag_maker function to 
# tag text responses with the appropriate
# part of speech and chunk tags. 
# This allows us to pull out specific parts-of-speech
# by different categories. For instance, adjectives may be 
# of interest by different wellbeing category.
# 
# by: Kimberly M. Kreiss
#####################################################
library(tidyverse)
library(parallel)
library(tidytext)
library(tm)
library(wordcloud)
source("tag_maker.R")

B2a <- read.csv('shed2018.csv') %>%
  filter(B2a_Refused != "Refused") %>% 
  mutate(B2a = as.character(B2a)) %>%
  select(B2a, B2, CaseID)

# because of reliance on rJava, this code takes a long time to run. 
# use parallelization to make it run quicker. 
# Calculate the number of cores
no_cores <- detectCores()

# Initiate cluster
cl <- makeCluster(no_cores)

clusterEvalQ(cl, {
  library(tidyverse)
  library(rJava)
  library(openNLP)
  library(tidyverse)
  library(NLP)
  library(openNLPmodels.en)
  library(tm)
  library(stringr)
  library(gsubfn)
  library(SentimentAnalysis)
  library(snow)
  library(parallel)
})

B2a<-B2a %>% 
  mutate(index = as.character(CaseID)) %>% 
  select(-CaseID)

k <- as.list(B2a$B2a)

tp <- clusterMap(cl, tag_maker, k, B2a$index)
B2a_tagged <- data.frame(matrix(unlist(tp), nrow=10445, byrow=T)) %>% 
  rename(pos_tags=X1, chunk_tags=X2, index=X3) 

# merge back in the response and other additional info such as B2
B2a_tagged <- B2a_tagged %>% 
  full_join(B2a, by="index")

# Now that the responses are tagged, 
# create a dataframe of adjectives
# create a function that will extract adjectives from responses 
adjective_maker <- function(df){
  words <- list()
  titles <- list()
  for (i in 1:nrow(df)){
    words <- c(words,as.vector(str_split(df$B2a[i], " ")))
    titles <- c(titles,as.vector(str_split(df$pos_tags[i], " ")))
    
    names(titles[[i]]) <- words[[i]]
  }
  
  adjectives <- list()
  
  # extract adjectives, which are tagged with "JJ", "JJR", or "JJS"
  for (i in 1:nrow(df)){
    list_of_adjectives <- names(titles[[i]][titles[[i]]=="JJ" | titles[[i]]=="JJR" | titles[[i]]=="JJS"])
    adjectives[[i]] <- list_of_adjectives
  }
  return(adjectives)
}

# Now, let's look at the adjectives that are in the bottom two categories and the two top categories
# start with those not doing okay
not_okay <- B2a_tagged %>% 
  filter(B2 %in% c("Just getting by", "Finding it difficult to get by"))

not_okay_adj <- adjective_maker(not_okay)
not_okay_adj_df <- as.data.frame(unlist(not_okay_adj)) %>%
  rename(word=`unlist(not_okay_adj)`) %>%
  mutate(word = as.character(word)) 

# let's plot them on a graph and analyze sentiment 
# using a lexicon based approach. Note that this will 
# only show words that show up in the lexicon
tokens <- not_okay_adj_df %>% 
  unnest_tokens(word, word)
tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("tomato2", "chartreuse3"))

## doing okay
doing_okay <- B2a_tagged %>% 
  filter(B2 %in% c("Doing okay", "Living comfortably"))

doing_okay_adj <- adjective_maker(doing_okay)
doing_okay_adj_df <- as.data.frame(unlist(doing_okay_adj)) %>%
  rename(word=`unlist(doing_okay_adj)`) %>%
  mutate(word = as.character(word)) 

# let's plot them on a graph and analyze sentiment 
# using a lexicon based approach. Note that this will 
# only show words that show up in the lexicon
tokens <- doing_okay_adj_df %>% 
  unnest_tokens(word, word)
tokens %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("tomato2", "chartreuse3"))
