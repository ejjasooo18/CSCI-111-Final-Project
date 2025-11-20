import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


file_name = "winequality-red.csv"
threshold = 7           
# Threshold for separating "Good" and "Bad" wine according to its quality
# Good (1) if quality >= 7; Bad (0) if quality < 7
# This was done to simplify the prediction of the quality

# --- Preparing the dataset --- #
df = pd.read_csv(file_name, sep=";") # loads the file into a DataFrame
print("Dataset loaded:", file_name)
print(df.head()) # prints the first 5 rows to verify that the file loaded correctly

# creates another column for quality_binary based on the wine's quality and threshold
df["quality_binary"] = (df["quality"] >= threshold).astype(int) 

# splitting data into X and y
# X: features used to make predictions (Acidity, Sugar, pH, etc.)
# y: data we want to predict (quality_binary) 
X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]

# splitting data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling the wine data in X so that they generally fall between -1 and 1.
# this makes the model understand the data better.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# prints the final shapes
print("\nFinal Shapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --- MODEL 1: Logistic Regression --- #
print("\n--- Training Model 1: Logistic Regression ---")
log_reg = LogisticRegression(random_state=42)   # initializes the model
log_reg.fit(X_train_scaled, y_train)    # model studies the data

y_pred_log = log_reg.predict(X_test_scaled) # model predicts quality based on wine's features using Logistic Regression
acc_log = accuracy_score(y_test, y_pred_log)    # gets the accuracy score

# prints the results
print(f"Logistic Regression Accuracy: {acc_log:.2%}")
print("\nClassification Report (LogReg):")
print(classification_report(y_test, y_pred_log))


# --- MODEL 2: Random Forest --- #
print("\n--- Training Model 2: Random Forest ---")
# n_estimators=100 means we create 100 decision trees and average them
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)    # initializes the model
rf_model.fit(X_train_scaled, y_train)   # model studies the data

y_pred_rf = rf_model.predict(X_test_scaled) # model predicts quality based on wine's features using Random Forest
acc_rf = accuracy_score(y_test, y_pred_rf)  # gets the accuracy score

# prints the results
print(f"Random Forest Accuracy: {acc_rf:.2%}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# --- VISUALIZATION: Confusion Matrix Comparison ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# plot Logistic Regression Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Logistic Regression (Acc: {acc_log:.2f})')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# plot Random Forest Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Random Forest (Acc: {acc_rf:.2f})')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show() # this will pop up a window with the charts
