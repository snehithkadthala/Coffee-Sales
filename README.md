# Coffee-Sales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('index.csv')
print(data.head())

print(data.isnull().sum())

# Fill missing categorical values with the mode
data['card'].fillna(data['card'].mode()[0], inplace=True)
print("Missing values after imputation:")
print(data.isna().sum())

print("First few rows of the DataFrame:")
print(data.head())

# Convert Date to datetime type
data['date'] = pd.to_datetime(data['date'])
# Check the data types
print(data.dtypes)

# Remove outliers based on Z-score
from scipy.stats import zscore
data = data[(np.abs(zscore(data[['money']])) < 3).all(axis=1)]

data['Month'] = data['date'].dt.month
data['Year'] = data['date'].dt.year

data.drop(columns=['date'], inplace=True)

print(data)

#EDA

plt.figure(figsize=(14,6))
sns.barplot(data=data, x='coffee_name', y='money')
plt.xlabel('Name')
plt.ylabel('Price')
plt.title('Coffee Price')
plt.show()

#Purchase
# Assuming `data` is a pandas DataFrame
# Count occurrences of 'coffee_name'
Y = data['coffee_name'].value_counts().reset_index()
Y.columns = ['coffee_name', 'count']
data['datetime'] = pd.to_datetime(data['datetime'])


plt.figure(figsize=(14, 6))
sns.barplot(data=Y, x='coffee_name', y='count')
plt.xlabel('Name')
plt.ylabel('Number of Purchases')
plt.title('Purchase Distribution by Coffee Name')
plt.show()


# Extract hour from datetime
data['hour'] = data['datetime'].dt.hour

# Plot sales distribution by hour of the day
plt.figure(figsize=(14, 7))
sns.histplot(data=data, x='hour', kde=True, bins=24, color='#0000ff')
plt.title('Sales Distribution by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Sales Count')
plt.show()

#CardvsCash
data = data['cash_type'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Cash Type Distribution')
plt.show()

print(data)

#Machine Learning

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('index.csv')
print(data.head())

print(data.isnull().sum())

# Fill missing categorical values with the mode
data['card'].fillna(data['card'].mode()[0], inplace=True)
print("Missing values after imputation:")
print(data.isna().sum())

print("First few rows of the DataFrame:")
print(data.head())

# Convert Date to datetime type
data['date'] = pd.to_datetime(data['date'])
# Check the data types
print(data.dtypes)

# Remove outliers based on Z-score
from scipy.stats import zscore
data = data[(np.abs(zscore(data[['money']])) < 3).all(axis=1)]

data['Month'] = data['date'].dt.month
data['Year'] = data['date'].dt.year

data.drop(columns=['date'], inplace=True)

print(data)

# Define features and target variable
X = data.drop(columns=['money'])
y = data['money']

# Handle missing values if any
X = X.fillna(0)  # Replace NaN with 0 or use other imputation methods as needed

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot Predictions vs Actual Values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.grid(True)
plt.show()

# Plot Residuals
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

# Plot Feature Coefficients (Absolute Values for importance)


