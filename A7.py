# Perform the following operations using Python on any open source dataset
# 1.Import all the required Python Libraries.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 2. Locate open source data from the web (e.g. https://www.kaggle.com).
# Provide a clear description of the data and its source (i.e., URL of the web site).
# train.csv - Titanic - Machine Learning from Disaster
# url https://www.kaggle.com/c/titanic
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
# Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

# 3. Load the Dataset into the pandas data frame.
df = pd.read_csv('train1.csv')
df.head()

# 4. Display the initial statistics.
df.info()
df.describe()

# 5. Scan all variables for missing values and inconsistencies. 
# If there are missing values and/or inconsistencies, use any of the suitable techniques to deal with them.
df.isnull().sum()
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 6. Scan all numeric variables for outliers. 
# If there are outliers, use any of the suitable techniques to deal with them.
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lowerBound) | (df[col] > upperBound)]
    
    if not outliers.empty:
        print(f"Outliers in '{col}': ")
        print(outliers[[col]])
        print("\n")

# 7. Apply data transformations on at least one of the variables.    
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
df.head()

# 8. Turn categorical variables into quantitative variables in Python.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)
df.head()

# Save the modified DataFrame
df.to_csv('modified_titanic_dataset.csv', index=False)