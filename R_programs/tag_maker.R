library(tidyverse)
library(rJava)
library(openNLP)
library(tidyverse)
library(NLP)
library(openNLPmodels.en)
library(tm)
library(stringr)
library(gsubfn)
library(readstata13)
library(SentimentAnalysis)
library(snow)
library(parallel)

# This program creates the tag_maker function, which reads in a string, its position in a dataframe, 
# and outputs a string with the appropriate part-of-speech tags. The part-of-speech tagger uses the 
# Apache OpenNLP library for tokenization, part of speech tagging, and chunking 
tag_maker <- function(s, index){
  require(openNLP)
  require(openNLPmodels.en)
  
  s <- as.String(s)
  
  ## Need sentence, word, and POS token annotations.
  sent_token_annotator <- Maxent_Sent_Token_Annotator()
  word_token_annotator <- Maxent_Word_Token_Annotator()
  pos_tag_annotator <- Maxent_POS_Tag_Annotator()
  
  ## Chunking needs word token annotations with POS tags.
  a3 <- annotate(s,
                 list(sent_token_annotator,
                      word_token_annotator,
                      pos_tag_annotator)) %>% 
    annotate(s, Maxent_Chunk_Annotator(), .) %>% 
    #pull out words
    subset(type == "word")
  # Extract the POS  from the words

  pos_tags <- purrr::map_chr(1:length(a3$features),function(i) {
    a3$features[[i]]$POS
    }) %>% paste(collapse=" ")
  chunk_tags <- purrr::map_chr(1:length(a3$features),function(i) {
    a3$features[[i]]$chunk_tag
    }) %>% paste(collapse=" ")
  
  gc()

  data_frame(pos_tags=pos_tags, chunk_tags=chunk_tags, index=index)
  
}
