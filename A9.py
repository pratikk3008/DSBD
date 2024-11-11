# Create a Linear Regression Model using Python/R to predict home
# prices using Boston Housing Dataset (https://www.kaggle.com/c/bostonhousing).
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Data
data = {
	'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
	'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features and target variable
X = df[['Size']]  # Features (input)
y = df['Price']   # Target (output)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional, but useful for understanding performance)
y_pred = model.predict(X_test)
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Predict price for a new house size
new_size = 2000
predicted_price = model.predict([[new_size]])  # Use the trained model to predict
print(f"Predicted price for a {new_size} sqft house: ${predicted_price[0]:.2f}")

# Save the trained model (optional)
joblib.dump(model, 'house_price_model.pkl')