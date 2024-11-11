# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

# Step 2: Load the dataset
data = pd.read_csv("es.csv")

# Step 3: Preprocess the data
# Drop 'Email No.' column and any unnecessary columns
data = data.drop('Email No.', axis=1)

# Define features (X) and target (y)
X = data.drop('Prediction', axis=1)  # 'Prediction' is the target (spam/not spam)
y = data['Prediction']

# Step 4: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train an SVM model
SVM = SVC(gamma='auto')  # Use 'auto' gamma for simplicity
SVM.fit(X_train, y_train)

# Step 6: Evaluate model performance
# Predictions on test set
y_pred = SVM.predict(X_test)

# Accuracy, precision, recall, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Step 7: Display results
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

# Classification report
print("Classification Report: ")
print(classification_report(y_test, y_pred))