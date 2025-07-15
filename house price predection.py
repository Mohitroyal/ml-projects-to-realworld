import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv('Housing.csv')
df.drop(['furnishingstatus'],axis=1,inplace=True)
bineary = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
le = LabelEncoder()
for col in bineary:
    df[col] = le.fit_transform(df[col])
print(df.head())
# split the data
X_data = df.drop('price',axis=1)
y = df['price']
# scaling data for LinearRegression
sc = StandardScaler()
X_data = sc.fit_transform(X_data)
# for random forest i use unscaled data
X_unscaled = df.drop('price', axis=1)
# Split data for both models
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_data, y, test_size=0.2, random_state=42)
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(X_unscaled, y, test_size=0.2, random_state=42)
# now training linearregression models
model_li = LinearRegression()
model_li.fit(X_train_scaled,y_train_scaled)
yl_pred = model_li.predict(X_test_scaled)
print(yl_pred)
# now traning randomforest
model_rf = RandomForestRegressor()
model_rf.fit(X_train_unscaled,y_train_unscaled)
yr_pred = model_rf.predict(X_test_unscaled)
print(yr_pred)
def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    print(f"{model_name} R2 Score: {r2:.4f}")
    print(f"{model_name} RMSE: {rmse:.2f}")

evaluate(y_test_scaled, yl_pred, "Linear Regression")
evaluate(y_test_unscaled, yr_pred, "Random Forest")



