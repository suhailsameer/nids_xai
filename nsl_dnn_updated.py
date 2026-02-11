import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import shap
from lime import lime_tabular
from collections import Counter

np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# 1. Load Dataset
# -------------------------

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df = pd.read_csv("KDDTest+.txt", names=columns)

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()

for col in categorical_cols:
    combined = pd.concat([train_df[col], test_df[col]])
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

feature_names = X_train.columns

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 2. Model Builder
# -------------------------

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------
# 3. Train Baseline
# -------------------------

model = build_model(X_train_scaled.shape[1])

model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

y_pred_baseline = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()

baseline_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_baseline),
    "Precision": precision_score(y_test, y_pred_baseline),
    "Recall": recall_score(y_test, y_pred_baseline),
    "F1": f1_score(y_test, y_pred_baseline)
}

plt.figure(figsize=(8,6))
plt.bar(baseline_metrics.keys(), baseline_metrics.values())
plt.ylim(0,1)
plt.title("Baseline Model Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("train_1.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------
# 4. SHAP Global Importance
# -------------------------

background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
explainer = shap.Explainer(model.predict, background)
shap_values = explainer(X_test_scaled[:10])

shap_importance = np.abs(shap_values.values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": feature_names,
    "importance": shap_importance
}).sort_values(by="importance", ascending=False)

top_shap_features = shap_df.head(20)['feature'].tolist()

# -------------------------
# 5. LIME Multi-Sample
# -------------------------

def lime_predict_fn(x):
    attack_prob = model.predict(x)
    normal_prob = 1 - attack_prob
    return np.hstack((normal_prob, attack_prob))

explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=['Normal', 'Attack'],
    mode='classification'
)

lime_counter = Counter()
indices = np.random.choice(len(X_test_scaled), 100, replace=False)

for idx in indices:
    exp = explainer_lime.explain_instance(
        X_test_scaled[idx],
        lime_predict_fn,
        num_features=10
    )
    for feature, weight in exp.as_list():
        for col in feature_names:
            if col in feature:
                lime_counter[col] += 1
                break


top_lime_features = [f[0] for f in lime_counter.most_common(20)]

# -------------------------
# 6. Consensus Selection
# -------------------------

consensus_features = list(set(top_shap_features).intersection(set(top_lime_features)))

if len(consensus_features) < 10:
    consensus_features = list(set(top_shap_features + top_lime_features))

# -------------------------
# 7. Retrain Optimized Model
# -------------------------

X_train_reduced = train_df[consensus_features]
X_test_reduced = test_df[consensus_features]

scaler_reduced = StandardScaler()
X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)

optimized_model = build_model(len(consensus_features))

optimized_model.fit(
    X_train_reduced_scaled, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

y_pred_opt = (optimized_model.predict(X_test_reduced_scaled) > 0.5).astype(int).flatten()

optimized_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_opt),
    "Precision": precision_score(y_test, y_pred_opt),
    "Recall": recall_score(y_test, y_pred_opt),
    "F1": f1_score(y_test, y_pred_opt)
}

plt.figure(figsize=(8,6))
plt.bar(optimized_metrics.keys(), optimized_metrics.values())
plt.ylim(0,1)
plt.title("SHAP + LIME Optimized Model Metrics")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("train_2.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------
# 8. Print Comparison
# -------------------------

print("\nBaseline Metrics:", baseline_metrics)
print("Optimized Metrics:", optimized_metrics)
