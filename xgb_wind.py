import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay)

# Paths & Data Loading
TRAIN_PATH = "/home/youssef/Downloads/WindTrainDATA(1)/Train.csv"
TEST_PATH  = "/home/youssef/Downloads/WindTestDATA(1)/Test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)


#Imputation (Using train medians for consistency)
# We calculate medians from train_df only to prevent data leakage from test_df
v1_med = train_df["V1"].median(skipna=True)
v2_med = train_df["V2"].median(skipna=True)

for df in [train_df, test_df]:
    df["V1"] = df["V1"].fillna(v1_med)
    df["V2"] = df["V2"].fillna(v2_med)

# Data preparation
X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"].astype(int)
X_test  = test_df.drop(columns=["Target"])
y_test  = test_df["Target"].astype(int)

# Calculate scale_pos_weight to handle Class Imbalance (Ratio of Negatives to Positives)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Convert data into XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

# Hyperparameter Setup & Auto Optimization
params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": float(scale_pos_weight),
    "seed": 42,
}

print("Running Cross-Validation to find the optimal number of rounds...")

# Use xgb.cv to find the best num_boost_round automatically
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    nfold=5,
    early_stopping_rounds=50,
    metrics="aucpr",
    as_pandas=True,
    seed=42
)

best_num_rounds = cv_results.shape[0]
print(f"Optimal rounds found: {best_num_rounds}")

# Final Model Training
print(f"Training final model on full training set with {best_num_rounds} rounds...")
booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=best_num_rounds)

# Visualization: Feature Importance
importance = booster.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Gain': list(importance.values())
}).sort_values(by='Gain', ascending=False).head(20)

plt.figure(figsize=(12, 6))
plt.bar(importance_df['Feature'], importance_df['Gain'], color='skyblue', edgecolor='navy')
plt.ylabel('Importance Score (Gain)')
plt.xlabel('Sensors')
plt.title('Top 20 Features Influencing the Predictive Model')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Evaluation & Performance Metrics
p_test = booster.predict(dtest)
yhat_test = (p_test >= 0.5).astype(int)

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
    }

metrics = calculate_metrics(y_test, yhat_test)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, yhat_test, cmap='Blues', ax=ax)
plt.title("Confusion Matrix: Failure vs. Healthy Prediction")
plt.show()

# Plot Performance Metrics
plt.figure(figsize=(10, 5))
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
plt.ylim(0, 1.1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', fontweight='bold')
plt.title("Final Model Performance Evaluation")
plt.show()

# 8) Save Model for Deployment
MODEL_NAME = "wind_final_full_train.json"
booster.save_model(MODEL_NAME)
print(f"\nSuccess! Model trained for {best_num_rounds} rounds and saved as {MODEL_NAME}.")