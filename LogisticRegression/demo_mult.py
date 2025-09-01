import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mult_logistic_regression import MultinomialLogisticRegression


data_1 = pd.read_csv("iris_multinomial.csv")
data_1

data_1["species"] = data_1["species"].map({"Iris-setosa" : 0,"Iris-virginica" : 2,"Iris-versicolor" : 1})
data_1

X = data_1.iloc[:,:-1].values
y = data_1.iloc[:,-1].values



X_scaled = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)


model = MultinomialLogisticRegression(verbose=True)

self = model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)

y_pred_train = model.predict(X_train)
# print(y_pred_train)
# print(y_train)

y_wrong = abs(y_pred_train - y_train)
print(np.sum(y_wrong))


