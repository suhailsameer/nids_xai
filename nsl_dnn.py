# =========================
# NSL-KDD DNN + SHAP + LIME
# =========================

import numpy as np
import pandas as pd

# ML / DL
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Explainability
import shap
from lime import lime_tabular

# -------------------------
# 1. Load NSL-KDD Dataset
# -------------------------

# Column names for NSL-KDD
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

# -------------------------
# 2. Preprocessing
# -------------------------

# Binary classification: normal vs attack
train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop difficulty column
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

# Encode categorical features
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

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# 3. Build DNN Model
# -------------------------

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
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

# -------------------------
# 4. Train Model
# -------------------------

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# -------------------------
# 5. Evaluate Model
# -------------------------

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# 6. SHAP Explainability (Recommended)
# -------------------------

# Use a smaller background for performance
background = X_train[np.random.choice(X_train.shape[0], 200, replace=False)]

explainer = shap.Explainer(
    model,
    background,
    feature_names=train_df.drop('label', axis=1).columns
)

shap_values = explainer(X_test[:100])

shap.summary_plot(shap_values, X_test[:100])

# -------------------------
# 7. LIME Explainability
# -------------------------

def lime_predict_fn(x):
    """
    Convert sigmoid output:
    [P(attack)]
    into:
    [P(normal), P(attack)]
    """
    attack_prob = model.predict(x)
    normal_prob = 1 - attack_prob
    return np.hstack((normal_prob, attack_prob))


explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=train_df.drop('label', axis=1).columns,
    class_names=['Normal', 'Attack'],
    mode='classification'
)

# Explain one instance
i = 5
exp = explainer_lime.explain_instance(
    X_test[i],
    lime_predict_fn,
    num_features=10
)

# exp.show_in_notebook()
exp.save_to_file("lime_explanation.html")
