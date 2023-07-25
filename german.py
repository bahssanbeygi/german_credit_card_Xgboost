import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# Load and preprocess the dataset
data = pd.read_csv('german_credit_data.csv')
y = data['Purpose']
X = data.drop('Purpose', axis=1)
X = pd.get_dummies(X, drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost classifier
xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
xgb.fit(X_train, y_train)

# Generate predictions and evaluate the performance
y_pred = xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Train an Explainable Boosting Classifier (EBC)
ebc = ExplainableBoostingClassifier(random_state=42)
ebc.fit(X_train, y_train)

# Generate global explanations for the EBC
global_explanation = ebc.explain_global()
show(global_explanation)

# Generate local explanations for the EBC
local_explanation = ebc.explain_local(X_test)
show(local_explanation)
