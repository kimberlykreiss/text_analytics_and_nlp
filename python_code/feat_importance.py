"""
This program uses a random forest and logistic 
regression with l1 penalty (lasso) to find which 
words are most predictive of positive and negative 
financial wellbeing. I use bag of words to transform 
my text data into numeric data. 

by: Kimberly M. Kreiss
"""
# import custom made functions 
from functions import accuracy_metrics, feature_importance_graph, confusion_matrix_viz, lasso_coef_viz
#import appropriate packages 
import pandas as pd 
import numpy as np
import matplotlib 
import nltk 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import multiprocessing
import sklearn

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

########################################################################### 
#  Step 1: data cleaning
########################################################################### 
#Read in data
SHED = pd.read_csv("df.csv",encoding = "latin1", dtype="str")
#Create dataset for machine learning 
B2a= SHED[['B2a', 'B2', 'CaseID', 'B2a_Refused', 'I40']]
#filter out responses that were refused
B2a = B2a[B2a.B2a_Refused!="Refused"]
#filter out responses where someone answered but wrote nothing
B2a = B2a[B2a['B2a'].notnull()]
#keep everything except the B2a_Refused variable
B2a = B2a[['CaseID','B2', 'B2a', 'I40']]
#create label variable for ok or not 
B2a['not_okay'] = B2a['B2'].transform(lambda x: 1 if x in ("Just getting by","Finding it difficult to get by") else 0)
#just keep B2a and the outcome variable 
B2a = B2a[['B2a', 'not_okay', 'I40']]

#preprocessing of text data 
#to lower case
B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing punctuation
B2a['B2a'] = B2a['B2a'].str.replace('[^\w\s]','') # finds any character that is not a word or white space and replaces with ''
# Stop word removal
#stop = stopwords.words('english')
#B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Stemming
st = PorterStemmer()
B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

B2a['B2a'].head()

########################################################################### 
#  Step 2: feature transformation and training the model 
########################################################################### 
#do bag of words transofrmation and run thorugh algotirhm 
words = set(nltk.corpus.words.words())
# split the data into test and training
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(B2a['B2a'], B2a['not_okay'], test_size=.3, random_state=200)

# Initialize a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 900, max_depth = 100) 

# Initialize a logistic classifier 
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(penalty='l1') 

# Now need to transform the input data into something the model can handle. use bag of words 
vectorizer = CountVectorizer(analyzer = "word",  \
                             ngram_range = (2,2), \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 

vector = vectorizer.fit_transform(x_train) # transform the training features
train_data_features=vector.toarray() #make into array for forest.fit()
vector1 = vectorizer.transform(x_test) #transform the testing features
test_data_features=vector1.toarray() #change test features to numeric input

#train a random forest
rf = forest.fit(train_data_features, y_train) #train the model 
y_pred_rf = rf.predict(test_data_features) # use the model on the testing data set 
y_pred_score_rf = rf.predict_proba(test_data_features)


#train a logistic regression 
lasso = logistic.fit(train_data_features, y_train)
y_pred_lasso = lasso.predict(test_data_features)
y_pred_score_lasso = lasso.predict_proba(test_data_features)

### Accuracy metrics

print("Accuracy score using random forest:")
sklearn.metrics.accuracy_score(y_pred_rf, y_test)
print("Accuracy score using logistic LASSO regression:")
sklearn.metrics.accuracy_score(y_pred_lasso, y_test)

# Plot feature importances 
feature_importance_graph(forest)
lasso_coef_viz(logistic)
## Accuracy metrics 
accuracy_metrics(y_test, y_pred_rf, y_pred_score_rf)
accuracy_metrics(y_test, y_pred_lasso, y_pred_score_lasso)
#confusion matrix
class_names = B2a['not_okay'].unique() 
confusion_matrix_viz(y_test, y_pred_rf)
confusion_matrix_viz(y_test, y_pred_lasso)
