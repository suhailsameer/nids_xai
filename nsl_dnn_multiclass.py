import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import shap
from lime import lime_tabular

# 1. Load NSL-KDD Dataset
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
# 2. Multiclass Preprocessing
# -------------------------

# Define attack categories
attack_map = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}

train_df['label'] = train_df['label'].map(attack_map)
test_df['label'] = test_df['label'].map(attack_map)

# Encode Labels to integers
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])
num_classes = len(label_encoder.classes_)

# Drop difficulty and scale features as before
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X_train = train_df.drop('label', axis=1)
y_train = to_categorical(train_df['label'], num_classes) # One-hot encoding for training

X_test = test_df.drop('label', axis=1)
y_test = test_df['label'] # Keep as integers for evaluation

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 3. Updated DNN Model 
# -------------------------

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.fit(X_train_scaled, y_train, epochs=10, batch_size=256, validation_split=0.2, verbose=1)


# -------------------------
# 5. Evaluate & Generate Metrics 
# -------------------------

# Get predicted classes
y_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_probs, axis=1)

# Generate confusion matrix
# y_test should be the ground truth integer labels
confusion = confusion_matrix(y_test, y_pred)

class_names = label_encoder.classes_  
# New list of metrics
metrics_graph = ['Accuracy', 'Precision', 'Recall', 'F1', 'BACC', 'MCC']
metric_values = {class_name: [] for class_name in class_names}

print("\n--- Detailed Metrics per Class ---")

for i, class_name in enumerate(class_names):
    TP = confusion[i, i]
    FP = confusion[:, i].sum() - TP
    FN = confusion[i, :].sum() - TP
    TN = confusion.sum() - TP - FP - FN
    
    # Accuracy, Precision, Recall, F1 Score
    Acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    
    # Balanced Accuracy
    TPR = Recall
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    Balanced_accuracy = (TPR + TNR) / 2
    
    # Matthews Correlation Coefficient
    mcc_num = (TP * TN) - (FP * FN)
    mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Matthews = mcc_num / mcc_den if mcc_den > 0 else 0
    
    # Store metrics 
    metric_values[class_name] = [Acc, Precision, Recall, F1_score, Balanced_accuracy, Matthews]

    print(f"Metrics for {class_name:7}: Acc: {Acc:.4f}, F1: {F1_score:.4f}, MCC: {Matthews:.4f}")

# Plotting the Performance Graph 
data = np.array([metric_values[class_name] for class_name in class_names])
x = np.arange(len(metrics_graph)) 
width = 0.14  

fig, ax = plt.subplots(figsize=(14, 7))

for i, class_name in enumerate(class_names):
    ax.bar(x + i * width, data[i], width, label=class_name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('NSL-KDD Classification Performance (Per Class)')
ax.set_xticks(x + width * (len(class_names) - 1) / 2)
ax.set_xticklabels(metrics_graph)
ax.legend(title='Attack Categories', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.ylim(0, 1.1) 
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the metrics plot
plt.savefig('metrics_by_class.png', bbox_inches='tight', dpi=300)
plt.show()

# -------------------------
# 4.  SHAP Explainer for Multiclass
# -------------------------


y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# SHAP Explainer (DeepExplainer or KernelExplainer)
# For Multiclass, SHAP returns a list of arrays (one per class)
background = X_train_scaled[np.random.choice(X_train_scaled.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X_test_scaled[:20])

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled[:20], feature_names=X_train.columns, show=False)
plt.savefig("shap_summary_multiclass.png", bbox_inches='tight', dpi=300)
plt.close() 
print("\nSHAP summary plot saved as 'shap_summary_multiclass.png'")

# -------------------------
# 5. Updated LIME
# -------------------------


explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X_train.columns,
    class_names=label_encoder.classes_,
    mode='classification'
)

# Explain an instance for all classes
i = 5
exp = explainer_lime.explain_instance(
    X_test_scaled[i], 
    model.predict, 
    num_features=10,
    top_labels=1 
)

exp.save_to_file("lime_multiclass.html")