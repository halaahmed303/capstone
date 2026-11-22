import zipfile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

zip_path = r"c:\Users\هلا\OneDrive - MOE Stem Schools\Documents\cap\archive (1).zip"

with zipfile.ZipFile(zip_path, "r") as z:
  csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
  with z.open(csv_name) as f:
    df = pd.read_csv(f)

sort_cols = [c for c in ["baby_id", "age_days", "date"] if c in df.columns]
if sort_cols:
  df = df.sort_values(sort_cols).reset_index(drop=True)

base_features = [
    c
    for c in [
        "temperature_c",
        "heart_rate_bpm",
        "oxygen_saturation",
        "respiratory_rate_bpm",
        "age_days",
        "weight_kg",
    ]
    if c in df.columns
]

df_feat = df.copy()
group_col = "baby_id" if "baby_id" in df_feat.columns else None
seq_cols = [
    c
    for c in [
        "temperature_c",
        "heart_rate_bpm",
        "oxygen_saturation",
        "respiratory_rate_bpm",
    ]
    if c in df_feat.columns
]

for col in seq_cols:
  if group_col:
    df_feat[f"{col}_prev1"] = df_feat.groupby(group_col)[col].shift(1)
  else:
    df_feat[f"{col}_prev1"] = df_feat[col].shift(1)

all_features = base_features + [f"{c}_prev1" for c in seq_cols]

df_clean = df_feat[all_features + ["risk_level"]].dropna().copy()
df_clean["target"] = (
    df_clean["risk_level"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"healthy": 0, "at risk": 1})
)
df_clean = df_clean.dropna(subset=["target"]).copy()
df_clean["target"] = df_clean["target"].astype(int)

X = df_clean[all_features]
y = df_clean["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"نسبة الدقة (Accuracy): {acc * 100:.2f}%")
print("\nمصفوفة الارتباك (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\nتقرير التصنيف (Classification Report):")
print(
    classification_report(
        y_test, y_pred, target_names=["Healthy (0)", "At Risk (1)"]
    )
)