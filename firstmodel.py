import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

# Load the dataset and handle missing values
df = pd.read_csv("train.csv", usecols=["Age", "Fare", "Survived"])
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Define feature matrix 'x' and target vector 'y'
x = df.iloc[:, 1:3]
y = df.iloc[:, 2]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
clf = LogisticRegression()

# Fit the model to the training data
clf.fit(x_train, y_train)

# Predict on the test data
y_predict = clf.predict(x_test)

# Calculate and print the accuracy score without transformation
print("Accuracy score without transformation: ", accuracy_score(y_test, y_predict))

# Transform the features using log1p
transform = FunctionTransformer(np.log1p, validate=True)
x_train_transformed = transform.fit_transform(x_train)
x_test_transformed = transform.transform(x_test)  # Use transform here, not fit

# Initialize and fit the Logistic Regression model on transformed data
clf2 = LogisticRegression()
clf2.fit(x_train_transformed, y_train)
y_predict_transformed = clf2.predict(x_test_transformed)

# Calculate and print the accuracy score after transformation
print("Accuracy score after transformation: ", accuracy_score(y_test, y_predict_transformed))
