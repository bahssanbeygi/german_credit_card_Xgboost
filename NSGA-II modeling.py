import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# load data
data = pd.read_csv('german_credit_data.csv')

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Purpose', axis=1), data['Purpose'], test_size=0.2, random_state=0)

# resample data
sm = SMOTE(random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)

# train models
rf = RandomForestClassifier(n_estimators=100, random_state=0)
dt = DecisionTreeClassifier(random_state=0)
xgb = XGBClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
dt.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# evaluate models
print('Random Forest Accuracy:', accuracy_score(y_test, rf.predict(X_test)))
print('Decision Tree Accuracy:', accuracy_score(y_test, dt.predict(X_test)))
print('XGBoost Accuracy:', accuracy_score(y_test, xgb.predict(X_test)))

# plot results
plt.plot(rf.feature_importances_, label='Random Forest')
plt.plot(dt.feature_importances_, label='Decision Tree')
plt.plot(xgb.feature_importances_, label='XGBoost')
plt.legend()
plt.show()
