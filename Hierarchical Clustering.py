# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:50:07 2024

@author: Jash Patel
"""

# Importing the libraries  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# Importing the dataset  
dataset = pd.read_csv('Mall_Customers_data_corrected.csv')  # Ensure the path is correct

# Extracting the relevant columns for clustering
x = dataset.iloc[:, [3, 4]].values  # Assuming Annual Income is at index 3 and Spending Score at index 4

# Finding the optimal number of clusters using the dendrogram  
import scipy.cluster.hierarchy as shc  
plt.figure(figsize=(10, 7))
dendrogram = shc.dendrogram(shc.linkage(x, method="ward"))  
plt.title("Dendrogram Plot")  
plt.ylabel("Euclidean Distances")  
plt.xlabel("Customers")  
plt.show()  

# Training the hierarchical model on the dataset
from sklearn.cluster import AgglomerativeClustering  
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')  # Use 'metric' instead of 'affinity'
y_pred = hc.fit_predict(x)  

# Visualizing the clusters  
plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for i in range(5):
    plt.scatter(x[y_pred == i, 0], x[y_pred == i, 1], s = 100, c = colors[i], label = labels[i])

plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()
