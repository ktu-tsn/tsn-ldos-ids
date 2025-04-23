# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:42:46 2024

@author: Mustafa Topsakal
"""
import xgboost as xgb
import time
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 


scaler = StandardScaler()

ml_methods = [
    ("KNeighborsClassifier", lambda: KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", leaf_size=30)),
    ("DecisionTreeClassifier", lambda: DecisionTreeClassifier(criterion="gini", random_state=42, splitter='best', max_depth=None)),
    ("RandomForestClassifier", lambda: RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42, max_features="sqrt")),
    ("XGBoost", lambda: xgb.XGBClassifier(objective='multi:softmax', num_class=2, random_state=42)),
    ("LightGBM", lambda: lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, max_depth=-1, learning_rate=0.1, n_estimators=100, random_state=42)),
    ("SVC", lambda: SVC(C=1.0, kernel="rbf",degree=3))
]

def classify(all_data, all_labels, classifier_index):
    
    fold = 0
    avAccuracy = 0
    avPrecision = 0
    avRecall = 0
    avF1 = 0
    avgTrainingTime = 0
    avgTestTime = 0
    results = ""
    model = None
    
    try:
        model = ml_methods[classifier_index][1]()
    except IndexError:
        print("Invalid classifier id!")
        return   
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    
    for i, (train_index, test_index) in enumerate(kf.split(all_data, all_labels)):

        fold+=1
        
        X_train = scaler.fit_transform(all_data[train_index, :])
        X_test = scaler.transform(all_data[test_index, :])

        y_train = all_labels[train_index]
        y_test = all_labels[test_index]
        
        # Fit model
        training_start_time = time.perf_counter()

        model.fit(X_train, y_train)
    
        training_end_time = time.perf_counter()
        training_time = training_end_time - training_start_time
        
        avgTrainingTime += training_time

        #Prediction is being made
        test_start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        test_end_time = time.perf_counter()
        test_time = test_end_time - test_start_time
                
        avgTestTime += test_time

        #Show the number of 0 and 1 for each fold
        #print(f"Fold:{str(i+1)} - Number of Zeros:{np.sum(y_pred == 0)} - Number of Ones:{np.sum(y_pred == 1)}")
        
        # Evaluate the model's performance on the test data
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        avAccuracy += accuracy
        avPrecision += precision
        avRecall += recall
        avF1 += f1
        
    
    print("Average Accuracy:", avAccuracy / fold)
    print("Average Precision:", avPrecision / fold)
    print("Average Recall:", avRecall / fold)
    print("Average F1-Skor:", avF1 / fold)
    print("Average training time: " + str(avgTrainingTime / fold))
    print("Average test time: " + str(avgTestTime / fold))

    results += ("Model: " + str(model) + "\n" +
    "Avg. Accuracy: {}\n".format(avAccuracy / fold) + "Avg. Precision: {}\n".format(avPrecision / fold) +
    "Avg. Recall: {}\n".format(avRecall / fold) + "Avg. F1: {}\n".format(avF1 / fold) +
    "Avg. Training Time: {}\n".format(str(avgTrainingTime / fold)) +
    "Avg. Test Time: {}\n".format(str(avgTestTime / fold)))
    
    return results