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
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score


#Read in data
B2a = pd.read_csv("df.csv", encoding = "latin1")
#filter out responses that were refused
B2a = B2a[B2a.B2a_Refused!=-1]
#filter out responses where someone answered but wrote nothing
B2a = B2a[B2a['B2a'].notnull()]
#keep everything except the B2a_Refused variable
B2a = B2a.iloc[0:len(B2a),0:3]
#create label variable for ok or not 
B2a['not_okay'] = B2a['B2'].transform(lambda x: 1 if x in (1,2) else 0)
#just keep B2a and the outcome variable 
B2a = B2a[['B2a', 'not_okay']]

#preprocessing of text data 
#to lower case
B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing punctuation
B2a['B2a'] = B2a['B2a'].str.replace('[^\w\s]','') # finds any character that is not a word or white space and replaces with ''
# Stop word removal
stop = stopwords.words('english')
B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Stemming
st = PorterStemmer()
B2a['B2a'] = B2a['B2a'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

B2a['B2a'].head()
#create a random sample of responses
#note that I am doing this to ensure that 
#the sample does not have a specific order 
#this may change model accuracy each time it's run 
B2a = B2a.sample(n=10440)

#transforming words into feature vectors 
X_train = B2a.iloc[:7000, 0].values
y_train = B2a.iloc[:7000, 1].values

X_test = B2a.iloc[7000:, 0].values
y_test= B2a.iloc[7000:, 1].values

#distribution of training and test samples 
B2a['not_okay'].iloc[:7000].value_counts()
B2a['not_okay'].iloc[7000:].value_counts()

#do bag of words transofrmation and run thorugh algotirhm 
words = set(nltk.corpus.words.words())

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) #updating max features, making larger
                             
vector = vectorizer.fit_transform(X_train)
train_data_features=vector.toarray()
np.count_nonzero(train_data_features==1)
np.count_nonzero(train_data_features==0)

vocab = vectorizer.get_feature_names()
print(vocab)

dist = np.sum(train_data_features, axis=0)
#for tag, count in zip(vocab, dist):
 #   print (count, tag)


print( "Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 300 trees
forest = RandomForestClassifier(n_estimators = 300) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features,y_train )

vectorizer1 = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 
                             
vector1 = vectorizer1.fit_transform(X_test)
test_data_features=vector1.toarray()
test_data_features = test_data_features
y_pred=forest.predict(test_data_features)
y_pred_score = forest.predict_proba(test_data_features)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
importances = forest.feature_importances_
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
classification_report(y_test,y_pred)
result = forest.predict(test_data_features)


# plot feature importances
# get feature importances
importances = forest.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, vectorizer.get_feature_names())

# sort the array in descending order of the importances and select 100 most important
f_importances.sort_values(ascending=False, inplace=True)
f_importances[1:10]

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(121, 9), rot=90, fontsize=12)

# show the plot
plt.tight_layout()
plt.show()

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

#what's the distribution of our sample?
B2a['not_okay'].value_counts()



# confusion matrix for all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = B2a['not_okay'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)











