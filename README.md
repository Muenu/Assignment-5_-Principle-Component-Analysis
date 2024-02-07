# Assignment-5_-Principle-Component-Analysis

## 1. PCA Implementation
# Utilize PCA to demonstrate how essential variables can be acquired from the cancer dataset available from

from sklearn import datasets
 
# Load the Breast Cancer dataset

import pandas as pd

cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target

# Save to a CSV file
cancer_df.to_csv('breast_cancer_dataset.csv', index=False)

#Standardize the data
# Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales. It lets us continue with the transformation of the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms.
# StandardScaler will standardize the features by removing the mean and scaling to unit variance so that each feature has μ = 0 and σ = 1.(Reference: Geek for Geeks)

#Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Reading the file 
data=pd.read_csv(r'C:\\Users\\DORIS\\breast_cancer_dataset.csv')
data

#Data Exploratory

data.info()

data.isnull().sum()

#2.  Dimensionality Reduction:

# Reduce the dataset into 2 PCA components for the project.
# Since not all features are necessarily useful for the prediction. Therefore, we can remove those noisy features and make a faster model(Ref: Geek for Geeks and Chat GPT 3.5).

# Importing libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the Breast Cancer Wisconsin (Diagnostic) dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce the dataset into 2 components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualize the reduced dataset(Ref: Chat GPT 3.5)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', edgecolor='k')
plt.title('PCA - Breast Cancer Dataset (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target Class')
plt.show()

#2.  Dimensionality Reduction:

# Reduce the dataset into 2 PCA components for the project.
# Since not all features are necessarily useful for the prediction. Therefore, we can remove those noisy features and make a faster model(Ref: Geek for Geeks and Chat GPT 3.5).

# Importing libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the Breast Cancer Wisconsin (Diagnostic) dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to reduce the dataset into 2 components
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualize the reduced dataset(Ref: Chat GPT 3.5)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', edgecolor='k')
plt.title('PCA - Breast Cancer Dataset (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target Class')
plt.show()

# 3. Logistic Regression for prediction
# Loading libraries

logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

# Predictions on the test set
y_pred = logreg.predict(X_test_pca)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")

# The model has a 99% accuracy in testing for breast cancer.

