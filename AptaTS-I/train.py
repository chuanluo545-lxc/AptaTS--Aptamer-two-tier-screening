from natsort import natsorted
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle

def my_10_fold(my_dir, filename):
    train_x = []
    train_y = []

    train_data = pd.read_csv(os.path.join(my_dir, filename),encoding="gbk")
    train_x = train_data.iloc[:, 1:].values 
    train_y = train_data.iloc[:, 0].values
    
    clf = RandomForestClassifier()
    
    auc = cross_val_score(clf, train_x, train_y, cv=10, scoring='roc_auc')
    accuracy = cross_val_score(clf, train_x, train_y, cv=10, scoring='accuracy')
    f = cross_val_score(clf, train_x, train_y, cv=10, scoring='f1_weighted')
    precision = cross_val_score(clf, train_x, train_y, cv=10, scoring='precision')
    recall = cross_val_score(clf, train_x, train_y, cv=10, scoring='recall')

    print(f"10-fold result for {filename}:")
    print('auc', auc.mean())
    print('accuracy', accuracy.mean())
    print('f-score', f.mean())
    print('precision', precision.mean())
    print('recall', recall.mean())
    print('\n')

    clf.fit(train_x, train_y)
    model_filename = filename.replace('.csv', '.pkl')
    output_path = os.path.join(model_filename)
    with open(output_path, 'wb') as output:
        pickle.dump(clf, output)
    print(f"Model saved to {output_path}")

csv_path = "train.csv"
model_path = "model.pkl"
my_dir = os.path.dirname(csv_path)
filename = os.path.basename(csv_path)
my_10_fold(my_dir, filename, model_path)
