# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="830" height="1008" alt="image" src="https://github.com/user-attachments/assets/bd394e85-3239-465a-9971-3cda674bf9ad" />


## DESIGN STEPS

## STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

## STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

## STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

## STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.

## PROGRAM

### Name: S DINESH RAGHAVENDARA
### Register Number: 212224040078

```python
# -------------------- STEP 1 : UPLOAD DATASET -------------------- #
from google.colab import files
uploaded = files.upload()   # Upload customers.csv

# -------------------- STEP 2 : IMPORT LIBRARIES -------------------- #
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- STEP 3 : LOAD DATA -------------------- #
data = pd.read_csv("customers.csv")   # File uploaded from system

print("Dataset Preview:")
display(data.head())

# -------------------- STEP 4 : PREPROCESSING -------------------- #

# Drop ID column
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({
    "Work_Experience": 0,
    "Family_Size": data["Family_Size"].median()
}, inplace=True)

# Encode categorical columns
categorical_columns = [
    "Gender", "Ever_Married", "Graduated",
    "Profession", "Spending_Score", "Var_1"
]

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])

# -------------------- STEP 5 : SPLIT DATA -------------------- #
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# -------------------- STEP 6 : MODEL -------------------- #
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# -------------------- STEP 7 : TRAINING -------------------- #
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=100)

# -------------------- STEP 8 : EVALUATION -------------------- #
model.eval()
predictions, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)

class_report = classification_report(
    actuals,
    predictions,
    target_names=[str(i) for i in label_encoder.classes_]
)

print("\nName: S DINESH RAGHAVENDARA")
print("Register No: 212224040078")
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# -------------------- STEP 9 : CONFUSION MATRIX HEATMAP -------------------- #
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    fmt="g"
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# -------------------- STEP 10 : SAMPLE PREDICTION -------------------- #
sample_input = X_test[12].clone().unsqueeze(0).detach().type(torch.float32)

with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = label_encoder.inverse_transform(
        [predicted_class_index]
    )[0]

print("\nName: S DINESH RAGHAVENDARA")
print("Register No: 212224040078")
print(f"\nPredicted class: {predicted_class_label}")
print(f"Actual class: {label_encoder.inverse_transform([y_test[12].item()])[0]}")

```



## Dataset Information

<img width="1191" height="242" alt="image" src="https://github.com/user-attachments/assets/e36c0433-d3f0-494f-91d9-e621db7f076b" />


## OUTPUT

### Confusion Matrix

<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/335f177c-a4c0-4123-a531-317ded195298" />


### Classification Report

<img width="647" height="348" alt="image" src="https://github.com/user-attachments/assets/2122e8aa-24fd-4568-8cac-98eef6da3e4a" />

<img width="565" height="250" alt="image" src="https://github.com/user-attachments/assets/f6ec8b94-d590-4605-914a-b27d53fc0533" />


### New Sample Data Prediction



## RESULT
Thus neural network classification model is developded for the given dataset.
