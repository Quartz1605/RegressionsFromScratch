from src.linear_regression import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data_1 = pd.read_csv('New_stud_performance.csv')
data_1 = data_1.drop(columns='Unnamed: 0')

X = data_1.iloc[:,:-1].values
y = data_1.iloc[:,-1].values

X_scaled = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

model = LinearRegression(verbose=True)
check_self = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(r2_score(y_test,y_pred))
# achieved a r2_score of 0.9889833406641303



