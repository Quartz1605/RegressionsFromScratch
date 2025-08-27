from src.logistic_regression import LogisticRegression
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



log_reg = LogisticRegression(verbose=False)
check_self = log_reg.fit(X_train,y_train)


y_pred = log_reg.predict(X_test)


# loss_data = pd.Series(check_self.history)
# plt.plot(loss_data)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training-Loss Curve")
# plt.show()

# cm = confusion_matrix(y_test,y_pred)
# display = ConfusionMatrixDisplay(cm)
# display.plot(cmap="Blues")
# plt.show()

# y_proba = log_reg.predict_probab(X_test)  # implement predict_proba
# fpr, tpr, _ = roc_curve(y_test, y_proba)
# auc = roc_auc_score(y_test, y_proba)

# plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
# plt.plot([0,1], [0,1], 'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.show()









