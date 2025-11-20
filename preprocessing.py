import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


file_name = "winequality-red.csv"
threshold = 7           # Threshold for separating "Good" and "Bad" wine according to its quality


df = pd.read_csv(file_name, sep=";")
print("Dataset loaded:", file_name)
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df["quality_binary"] = (df["quality"] >= threshold).astype(int)

X = df.drop(["quality", "quality_binary"], axis=1)
y = df["quality_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFinal Shapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --- MODEL 1: Logistic Regression (The Baseline) ---
print("\n--- Training Model 1: Logistic Regression ---")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)

print(f"Logistic Regression Accuracy: {acc_log:.2%}")
print("\nClassification Report (LogReg):")
print(classification_report(y_test, y_pred_log))


# --- MODEL 2: Random Forest (The Powerhouse) ---
print("\n--- Training Model 2: Random Forest ---")
# n_estimators=100 means we create 100 decision trees and average them
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {acc_rf:.2%}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# --- VISUALIZATION: Confusion Matrix Comparison ---
# (This generates the chart Member 3 needs for the slides)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Logistic Regression Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Logistic Regression (Acc: {acc_log:.2f})')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Plot Random Forest Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Random Forest (Acc: {acc_rf:.2f})')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show() # This will pop up a window with the charts
