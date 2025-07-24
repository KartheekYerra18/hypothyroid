import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import recall_score


# Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/hypothyroid.csv")

# Encode categorical features
le = LabelEncoder()
for col in ["sex", "on thyroxine", "sick", "pregnant", "goitre", "tumor", "output"]:
    df[col] = le.fit_transform(df[col])

# Split dataset
X = df.drop(columns=["output"])  # Features
y = df["output"]  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

results = []  # Store model results

# Train & Evaluate Each Model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, f"models/{name}.pkl")

    # Store results
    results.append({"model": name, "accuracy": accuracy, "precision": precision, "f1_score": f1})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results as CSV for Flask to use
results_df.to_csv("model_results.csv", index=False)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(results_df["model"], results_df["accuracy"], color=["blue", "green", "red", "purple"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig("static/accuracy_chart.png")  # Save image for Flask
plt.close()

# Create a directory for confusion matrices if not exists
if not os.path.exists("static"):
    os.makedirs("static")

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

conf_matrices = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Save model
    joblib.dump(model, f"models/{name}.pkl")
    
    # Generate confusion matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    # Calculate precision, recall, and f1-score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    precision_scores[name] = precision
    recall_scores[name] = recall
    f1_scores[name] = f1


# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {name}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.savefig("static/confusion_matrix.png")  # Save the image
plt.show()
# Plot precision, recall, and f1-score graph
plt.figure(figsize=(8, 6))
plt.plot(models.keys(), precision_scores.values(), marker='o', linestyle='-', label="Precision")
plt.plot(models.keys(), recall_scores.values(), marker='s', linestyle='-', label="Recall")
plt.plot(models.keys(), f1_scores.values(), marker='^', linestyle='-', label="F1-Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.title("Model Performance: Precision, Recall, and F1-Score")
plt.legend()
plt.grid(True)
plt.savefig("static/performance_graph.png")  # Save the performance graph
plt.show()