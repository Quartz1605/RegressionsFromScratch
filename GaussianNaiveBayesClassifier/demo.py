from naive_bayes import GaussianNaiveBayes
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

data_1 = pd.read_csv("Breast-Cancer-Binary.csv")
data_1 = data_1.drop('Unnamed: 32',axis=1)
data_1 = data_1.drop('id',axis=1)
data_1['diagnosis'] = data_1['diagnosis'].map({"M" : 1,"B" : 0})

X = data_1.iloc[:,1:].values
y = data_1.iloc[:,0].values

scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

naive = GaussianNaiveBayes(verbose=True)

naive.fit(X_train,y_train)

y_pred = naive.predict(X_test)

print(y_test)
print(y_pred)

print(np.sum(abs(y_test-y_pred)))



