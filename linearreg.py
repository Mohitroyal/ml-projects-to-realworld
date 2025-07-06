import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("C:/Users/mpokk/Downloads/archive (4)/Study_vs_Score_data.csv")

print(df.columns)  # Check actual column names
print(df.head())

# Use correct column name (update if needed)
X_data = df.drop('Final_Marks', axis=1)
y = df['Final_Marks']

# Scale features
pre = StandardScaler()
x = pre.fit_transform(X_data)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=75)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)
coff = model.coef_
print(coff)
print(model.intercept_)
# Predict
pred = model.predict(X_test)


# Evaluate
r2 = r2_score(Y_test, pred)
print("R2 Score:", r2)
