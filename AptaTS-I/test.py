from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import cross_val_score
import pickle  
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(threshold=10000)
def load_and_test_model(pkl_path,test_data_path):
    with open(pkl_path, 'rb') as f:
        clf = pickle.load(f)

    test_data = pd.read_csv(test_data_path,encoding='gbk')
    test_x = test_data.iloc[:, 1:].values
    test_y = test_data.iloc[:, 0].values

    test_y_pred = clf.predict(test_x)
    test_y_proba = clf.predict_proba(test_x)[:, 1] 

    auc = roc_auc_score(test_y, test_y_proba)
    accuracy = accuracy_score(test_y, test_y_pred)
    f1 = f1_score(test_y, test_y_pred, average='weighted')
    precision = precision_score(test_y, test_y_pred)
    recall = recall_score(test_y, test_y_pred)
    cm = confusion_matrix(test_y,test_y_pred)
    print('Test results:')
    print('AUC:', auc)
    print('Accuracy:', accuracy)
    print('F1 Score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('prediction:',test_y_pred)

    print('Confusion Matrix:')
    print(cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(test_y_pred), yticklabels=np.unique(test_y))

    plt.title('Random Forest')
    plt.xlabel('Predicted ')
    plt.ylabel('True ')
    plt.show()

pkl_path = "model.pkl"
test_data_path = "test.csv"
load_and_test_model(pkl_path,test_data_path)    
