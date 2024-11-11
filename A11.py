import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\HP-PC\Downloads\archive\sales_data_sample.csv", encoding='latin')

# Display first few rows of the dataset
print(df.head())  

# Check the data types of columns
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())
df.dropna(inplace=True)  # Remove rows with missing values

# Selecting columns for clustering (using actual column names)
X = df[['QUANTITYORDERED', 'SALES']].values  # Replace with desired feature columns

# Initializing WCSS (within-cluster sum of squares)
wcss = []   

# Applying K-Means for different k values to use the elbow method
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plotting the Elbow Method graph
ks = list(range(1, 11))
plt.plot(ks, wcss, 'bx-')
plt.title("Elbow Method")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()  # Display the plot

# Statistical summary of the dataset
print(df.describe())

# Feature scaling
ss = StandardScaler()
scaled = ss.fit_transform(X)

# Recalculating WCSS for scaled data
wcss = []
for i in range(1, 11):
    clustering = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    clustering.fit(scaled)
    wcss.append(clustering.inertia_)
    
# Plotting the Elbow Method graph for scaled data
plt.plot(ks, wcss, 'bx-')
plt.title("Elbow Method (Scaled Data)")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()  # Display the plot

# Fit with the optimal number of clusters (replace optimal_k with the identified optimal value)
optimal_k = 4  # Example optimal value; adjust based on elbow method results
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(scaled)

# Save the model
joblib.dump(kmeans, 'kmeans_model.pkl')

# Visualizing the clusters
plt.scatter(scaled[:, 0], scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('Clusters and Centroids')
plt.xlabel('QUANTITYORDERED (scaled)')
plt.ylabel('SALES (scaled)')
plt.grid(True)
plt.show()