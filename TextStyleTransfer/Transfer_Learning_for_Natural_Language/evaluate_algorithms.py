import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time
from sklearn.svm import SVC # Support Vector Classification model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import csv

LOGISTIC_REGRESSION=False
SVC_MODEL=False
RANDOM_FOREST=False
GBM=True
SHOW_ROC_CURVE=True

# Load datasets
with open('./data/spam_detection/train_data.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f)

with open('./data/spam_detection/test_data.pkl', 'rb') as f:
    test_x, test_y = pickle.load(f)

# Define the file name
predictors_csv_file = "./data/spam_detection/predictors.csv"

# Read 'predictors' list from the CSV file
with open(predictors_csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        predictors = row
    

if LOGISTIC_REGRESSION: 
    start_time = time.time()
    def fit(train_x,train_y):
        model = LogisticRegression()
    
        try:
            model.fit(train_x, train_y)
        except:
            pass
        return model

    model = fit(train_x,train_y)
    end_time = time.time()
    print("Training the Logic Regression Classifier took %3.4d seconds"%(end_time-start_time))  
    predicted_labels = model.predict(test_x)
    from sklearn.metrics import accuracy_score
    acc_score = accuracy_score(test_y, predicted_labels)
    print(f"The logistic regression accuracy score is: {acc_score:.4f}" )
    
    


if SVC_MODEL:  
    start_time = time.time()
    clf = SVC(C=1, gamma="auto", kernel='linear',probability=False)
    clf.fit(train_x, train_y)
    end_time = time.time()
    print("Training the SVC Classifier took %3.4d seconds"%(end_time-start_time))    
    predicted_labels = clf.predict(test_x)
    acc_score = accuracy_score(test_y, predicted_labels)
    print(f"The SVC Classifier testing accuracy score is: {acc_score:.4f}")


if RANDOM_FOREST:
    clf = RandomForestClassifier(n_jobs=1, random_state=0)
    start_time = time.time()
    clf.fit(train_x, train_y)
    end_time = time.time()
    print("Training the Random Forest Classifier took %3.4d seconds"%(end_time-start_time))
    predicted_labels = clf.predict(test_x)    
    acc_score = accuracy_score(test_y, predicted_labels)
    print(f"The RF testing accuracy score is: {acc_score:.4f}")


if GBM:
    def modelfit(alg, train_x, train_y,  test_x, performCV=True, cv_folds=5):
        alg.fit(train_x, train_y)
        predictions = alg.predict(train_x)
        predprob = alg.predict_proba(train_x)[:,1]
        if performCV:
            cv_score = cross_val_score(alg, train_x, train_y, cv=cv_folds, scoring='roc_auc')
        
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(train_y,predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, predprob))
        if performCV:
            print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % 
    (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        print(f"predictors = {len(predictors)}")
        print(f"Importance Features = {len(alg.feature_importances_)}")    
        # An important property of decision-tree-based methods is that they can provide features an importance score, 
        # which can be used to detect the most important features in a given dataset.  
        # We can plot these and identify the most prominent features  
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False) 
        feat_imp[:10].plot(kind='bar',title='Feature Importances')
        return alg.predict(test_x),alg.predict_proba(test_x)
    
    gbm0 = GradientBoostingClassifier(random_state=10)
    start_time = time.time()
    test_predictions, test_probs = modelfit(gbm0, train_x, train_y,  test_x)
    end_time = time.time()
    print("Training the Gradient Boosting Classifier took %3.4d seconds"%(end_time-start_time))
    predicted_labels = test_predictions
    acc_score = accuracy_score(test_y, predicted_labels)
    print(f"The Gradient Boosting testing accuracy score is: {acc_score:.4f}")
    if SHOW_ROC_CURVE:
        # The straight line with a slope of 1 represents the FPR-versus-TPR trade-off corresponding to random chance. 
        # The further to the left the ROC curve is from this line, the better performing the classifier. 
        # As a result, the area under the ROC curve can be used as a measure of performance.
        test_probs_max = []
        for i in range(test_probs.shape[0]):
            test_probs_max.append(test_probs[i,test_y[i]])
        
        fpr, tpr, thresholds = metrics.roc_curve(test_y, np.array(test_probs_max))
        fig,ax = plt.subplots()
        plt.plot(fpr,tpr,label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Email Example')
        plt.legend(loc="lower right")
        plt.show()



 
