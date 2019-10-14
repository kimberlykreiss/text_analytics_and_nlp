##########################################################################
#
#    This program uses some basic text mining strategies to explore 
#    the responses to B2a. Output can be found in Appendix of GASP slides, 
# here: https://kimberlykreiss.github.io/projects_and_code/GASP_slides.pdf
# Much of this analysis follows Julia Silge's Tidy Text Mining with R 
# at https://tidytextmining.com
#
#       by: Kimberly M. Kreiss
#
##########################################################################
library(tidyverse)
library(igraph)
library(tidytext)
library(ggraph)
library(tm)

# read in SHED data
SHED <- read.csv('shed2018.csv')

#create a subset of SHED data and do appropriate cleaning
B2a_viz <- SHED %>% 
  select(CaseID, weight2b, B2, B2a, B2a_Refused) %>% 
  mutate(not_okay = if_else(B2a == "Finding it difficult to get by" | B2a == "Struggling to get by", 1, 0), 
         B2a = as.character(B2a)) %>%
  filter(B2a_Refused != "Refused")

#tokenize responses
tokens <- B2a_viz %>% 
  unnest_tokens(word, B2a)

#take out stop words 
#data("stop_words")
stop_words <- as.data.frame(stopwords("en")) %>% 
  rename(word = `stopwords("en")`)
tokens <- tokens %>%
  anti_join(stop_words)

# generate word frequencies by wellbeing category
word_frequencies <- tokens %>%
  group_by(B2) %>%
  count(word, sort = TRUE) %>%
  mutate(good_ranks = order(order(n, decreasing=TRUE))) %>%
  ungroup()

#the goal is to create a visualization showing the ten most common words 
#within each category, and how the frequency differs by wellbeing category
#identify the top 10 most common words for each wellbeing category 
words_of_interest <- word_frequencies %>% 
  filter(good_ranks < 10 ) %>% 
  ungroup() %>%
  select(word) %>%
  unique()

num_responses <- B2a_viz %>% 
  group_by(B2) %>% 
  summarise(num_responses_pergroup = n())# %>% 
# spread(key = B2, value = n)

# generate shares of top words for each category 
# how do they differ across wellbeing? 
# For example, around 12% of those who said they were 
# "Finding it difficult to get by" mentioned the word "Work", 
# compared with around 3% for those living comfortably
word_Freq_viz <- word_frequencies %>% 
  left_join(num_responses, by = "B2") %>% 
  mutate(share_ = n/num_responses_pergroup) %>%
  filter(word %in% words_of_interest$word ) %>%
  #  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, share_, fill=word)) +
  geom_col(fill = "cornflowerblue") +
  xlab(NULL) +
  coord_flip() + 
  theme(legend.position="none") +
  labs(title = "Word Frequencies in Write-in Responses", y = "Share of responses") + 
  theme(legend.position="none") + 
  facet_wrap(~B2) + 
  theme(panel.background = element_blank(), 
        panel.grid.major.x = element_line("black")) + 
  scale_y_continuous(labels = scales::percent)
word_Freq_viz

## overall word frequencies
word_frequencies_all<- tokens %>%
  count(word, sort = TRUE) %>%
  filter(n > 400) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill=word)) +
  geom_col(fill = "cornflowerblue") +
  xlab(NULL) +
  coord_flip() + 
  theme(legend.position="none") +
  labs(title = "Word Frequencies in Write-in Responses", y = "Count") + 
  theme(legend.position="none") + 
  theme(panel.background = element_blank(), 
        panel.grid.major.x = element_line("black"))
word_frequencies_all

######################################################################################
### bigram network analysis 
#   Word frequencies are interesting but provide limited information. An extension is
#   to observe word networks of different n-grams. For this analysis I choose bigrams. 
######################################################################################
# generate bigrams, separate them into two columns, and filter out bigrams where one word 
# is a stop word. 
B2a_bigrams <- B2a_viz  %>%
  unnest_tokens(bigram, B2a, token = "ngrams", n = 2)

bigrams_separated <- B2a_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

# Generate the network graph 
bigram_graph <- bigram_counts %>%
  filter(n > 30) %>%
  filter(!is.na(word1), !is.na(word2)) %>%
  graph_from_data_frame()

# a nicer graph 
set.seed(2016)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

network_graph <- ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()
network_graph
