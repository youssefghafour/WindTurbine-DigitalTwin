import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ==========================================
# 1) Paths & Loading
# ==========================================
TRAIN_PATH = "/home/youssef/Downloads/WindTrainDATA(1)/Train.csv"
TEST_PATH  = "/home/youssef/Downloads/WindTestDATA(1)/Test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# ==========================================
# 2) Imputation (Train Medians Only - No Leakage)
# ==========================================
v1_med = train_df["V1"].median(skipna=True)
v2_med = train_df["V2"].median(skipna=True)

for df in [train_df, test_df]:
    df["V1"] = df["V1"].fillna(v1_med)
    df["V2"] = df["V2"].fillna(v2_med)

# ==========================================
# 3) Prepare Data
# ==========================================
X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"].astype(int)

X_test  = test_df.drop(columns=["Target"])
y_test  = test_df["Target"].astype(int)

# معالجة اختلال الفئات (Class Imbalance)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ==========================================
# 4) Training on ALL Data
# ==========================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test, label=y_test)

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

# استخدام عدد الجولات الذي حددناه سابقاً لضمان أفضل دقة
num_rounds = 882

print(f"Training on FULL data for {num_rounds} rounds...")
booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)

# ==========================================
# 5) Visualization: Feature Importance (Top 20)
# ==========================================
importance = booster.get_score(importance_type='gain')

importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Gain': list(importance.values())
})

# ترتيب واختيار أفضل 20 ميزة
importance_df = importance_df.sort_values(by='Gain', ascending=False).head(20)

plt.figure(figsize=(15, 7))
# الرسم الرأسي (الميزات في الأسفل)
plt.bar(importance_df['Feature'], importance_df['Gain'], color='skyblue', edgecolor='navy')

plt.ylabel('Importance Score (Gain)')
plt.xlabel('Features (Sensors)')
plt.title('Top 20 Features Influencing the Model')
plt.xticks(rotation=45, ha='right') # تدوير الأسماء لتفادي التداخل
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ==========================================
# 6) Evaluation & Predictions
# ==========================================
p_test = booster.predict(dtest)
yhat_test = (p_test >= 0.5).astype(int)

def get_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
    }

m = get_metrics(y_test, yhat_test)

# Confusion Matrix
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, yhat_test, cmap='Blues')
plt.title("Confusion Matrix (XGBoost)")
plt.show()

# Performance Metrics Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(m.keys(), m.values(), color=['navy', 'darkred', 'darkgreen', 'orange'])
plt.ylim(0, 1.1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}", ha='center', va='bottom')
plt.title("Final Performance Metrics on Test Set")
plt.show()

# ==========================================
# 7) Save Model
# ==========================================
booster.save_model("wind_final_full_train.json")
print("\nDone! Model trained on 100% data, charts generated, and model saved.")