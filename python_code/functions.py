"""
This program creates custom made functions that generate 
accuracy metrics and plots for feature importance for 
random forest and lasso regression to be used with 
feat_importance.py. This program will allow for 
multiple models to be assessed and greatly improve readability 
of code. 

by: Kimberly M. Kreiss
"""
import pandas as pd 

def accuracy_metrics(y_test, y_pred, y_pred_score):
    from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score

    print("\n")
    print("Results Using All Features: \n")

    print("Classification Report: ")
    print(classification_report(y_test,y_pred))
    print("\n")

    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print("\n")

    print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)
    return;
    
def feature_importance_graph(model):
    importances = model.feature_importances_

    # convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
    f_importances = pd.Series(importances, vectorizer.get_feature_names())
    
    # sort the array in descending order of the importances and select 100 most important
    f_importances.sort_values(ascending=False, inplace=True)
    f_importances[1:10]
    
    # make the bar Plot from f_importances
    f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(121, 9), rot=90, fontsize=12)
    
    #export the plot to pdf 
    #plt.savefig('feat_importance.pdf')
    
    # show the plot
    plt.tight_layout()
    plt.show()
    return; 
    
def confusion_matrix_viz(y_test, y_pred): 
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
    classification_report(y_test,y_pred)
    #result = forest.predict(test_data_features)
    
    # confusion matrix for all features
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
    
    plt.figure(figsize=(5,5))
    
    hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    return;

def lasso_coef_viz(logistic_model): 
    feature_names = vectorizer.get_feature_names()

    coef_dict = {}
    for coef, feat in zip(logistic_model.coef_[0], feature_names):
        coef_dict[feat] = coef
    
    
    importances = pd.DataFrame([coef_dict]).T
    importances['features'] = importances.index
    importances['coef'] = importances.iloc[:,0]
    importances['index'] = range(500)
    importances = importances.iloc[:,1:4]
    importances = importances.set_index('index')
    importances.sort_values(by="coef",ascending=True, inplace=True)
    importances.plot(x='features', y='coef', kind='bar', figsize=(90, 9), rot=90, fontsize=12)
    #export the plot to pdf 
    #plt.savefig('feat_importance_logistic.pdf')
    
    # show the plot
    plt.tight_layout()
    plt.show()
    return;
