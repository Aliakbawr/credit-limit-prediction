import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

data = pd.read_csv('data/CreditPrediction.csv')

missing_percentages = data.isnull().mean()
column_to_delete = missing_percentages[missing_percentages > 0.9].index

data.drop(columns = ['Unnamed: 19', 'CLIENTNUM'], inplace = True)

# Calculate the mode (most frequent category) of the "Gender" column
mode_gender = data['Gender'].mode().iloc[0]

# Fill missing values with the mode
data['Gender'] = data['Gender'].fillna(mode_gender)

data['Marital_Status'] = data['Marital_Status'].fillna('Unknown')

mode_Card_Category = data['Card_Category'].mode()[0]
data['Card_Category'].fillna(mode_Card_Category, inplace=True)

data.dropna(subset=['Months_on_book'], inplace=True)
data.dropna(subset=['Total_Relationship_Count'], inplace=True)

# Calculate the mode (most frequent category) of the "Gender" column
mode_Education_Level = data['Education_Level'].mode().iloc[0]

data['Education_Level'] = data['Education_Level'].replace('Unknown', mode_Education_Level)

# Calculate the mode (most frequent category) of the "Gender" column
mode_Income_Category = data['Income_Category'].mode().iloc[0]

data['Income_Category'] = data['Income_Category'].replace('Unknown', mode_Income_Category)

data = pd.get_dummies(columns=['Gender','Marital_Status'], data=data)

Education_Level_map = {'High School': 1, 'Graduate': 3, 'Uneducated': 0,  'College': 2, 'Post-Graduate': 4,
 'Doctorate': 5}
data['Education_Level'] = data['Education_Level'].map(Education_Level_map)

Income_Category_map = {'$60K - $80K': 2, 'Less than $40K': 0, '$80K - $120K': 3, '$40K - $60K': 1, '$120K +': 4}
data['Income_Category'] = data['Income_Category'].map(Income_Category_map)

Card_Category_map = {'Blue':0, 'Gold': 2, 'Silver': 3, 'Platinum': 4}
data['Card_Category'] = data['Card_Category'].map(Card_Category_map)

limit = 100
count = np.count_nonzero(data['Customer_Age'] > limit)
data = data[data['Customer_Age'] < limit]

import pandas as pd

# Assuming you have a DataFrame 'df' with columns 'feature1', 'feature2', and 'feature3'
# Calculate IQR for each column
col_1 = 'Customer_Age'
col_2 =  'Total_Amt_Chng_Q4_Q1'
#'Months_on_book' #'Card_Category'
Q1 = data[[col_1, col_2]].quantile(0.25)
Q3 = data[[col_1, col_2]].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier boundaries for each column
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers for each column
data = data[
    (data[col_1] >= lower_bound[col_1]) & (data[col_1] <= upper_bound[col_1]) &
    (data[col_2] >= lower_bound[col_2]) & (data[col_2] <= upper_bound[col_2])
]

# Now 'filtered_df' contains the data without outliers for the specified columns
data.describe()

y = data['Credit_Limit']
data.drop(columns = ['Credit_Limit'], inplace = True)

data = (data - data.mean()) / data.std()
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, shuffle= True)

regr = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=10)
regr.fit(X_train, y_train)
predicts = regr.predict(X_test)
r2 = r2_score(y_test, predicts)
print("R-squared (R²) Score:", r2)
mse = mean_squared_error(y_test, predicts)
print("Mean Squared Error:", mse)

plt.scatter(y_test, predicts, color='blue', label='Predicted vs. True')
plt.plot([min(y_test), max(y_test)], [min(predicts), max(predicts)], color='red', linestyle='--', label='Predicted Line')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predicted vs. True Values')
plt.legend()
plt.show()