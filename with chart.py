import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv('german_credit_data.csv')
y = data['Purpose']
X = data.drop('Purpose', axis=1)
X = pd.get_dummies(X, drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

# Generate predictions and evaluate the performance
y_pred_nn = model.predict_step(X_test)
print('Neural Network Classifier:')
print(confusion_matrix(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Random Forest Classifier:')
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print('Decision Tree Classifier:')
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# One-Class SVM
svm_model = OneClassSVM(kernel='rbf', nu=0.1)
svm_model.fit(X_train[y_train==0])
y_pred_svm = svm_model.predict(X_test)
y_pred_svm[y_pred_svm == 1] = 0
y_pred_svm[y_pred_svm == -1] = 1
print('One-Class SVM:')
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Comparison Chart
results = pd.DataFrame({'Algorithm': ['Neural Network', 'Random Forest', 'Decision Tree', 'One-Class SVM'],
                        'Accuracy': [accuracy_score(y_test, y_pred_nn), accuracy_score(y_test, y_pred_rf), 
                                     accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_svm)],
                        'Precision': [precision_score(y_test, y_pred_nn,pos_label='positive'
                                           ,average='micro'), precision_score(y_test, y_pred_rf,pos_label='positive'
                                           ,average='micro'), 
                                      precision_score(y_test, y_pred_dt,pos_label='positive'
                                           ,average='micro'), precision_score(y_test, y_pred_svm,pos_label='positive'
                                           ,average='micro')],
                        'Recall': [recall_score(y_test, y_pred_nn,pos_label='positive'
                                           ,average='micro'), recall_score(y_test, y_pred_rf,pos_label='positive'
                                           ,average='micro'), 
                                   recall_score(y_test, y_pred_dt,pos_label='positive'
                                           ,average='micro'), recall_score(y_test, y_pred_svm,pos_label='positive'
                                           ,average='micro')],
                        'F1 Score': [f1_score(y_test, y_pred_nn,pos_label='positive'
                                           ,average='micro'), f1_score(y_test, y_pred_rf,pos_label='positive'
                                           ,average='micro'), 
                                     f1_score(y_test, y_pred_dt,pos_label='positive'
                                           ,average='micro'), f1_score(y_test, y_pred_svm,pos_label='positive'
                                           ,average='micro')]
                       })

print('\nComparison Chart:')
print(results)
fig, ax = plt.subplots(figsize=(8,6))
results.plot(kind='bar', x='Algorithm', y=['Accuracy', 'Precision', 'Recall', 'F1 Score'], ax=ax)
ax.set_xticklabels(results['Algorithm'], rotation=0)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.set_title('Comparison Chart')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Score')
plt.show()
