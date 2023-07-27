import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv('german_credit_data.csv')

# Define the target variable and features
target = df['Purpose']
features = df.drop(['Purpose'], axis=1)

# Convert string data to numerical data
le = LabelEncoder()
for col in features.select_dtypes(include=['object']):
    features[col] = le.fit_transform(features[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the hyperparameters to tune
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Define the model
lr = LogisticRegression()

# Define the GridSearchCV object
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and performance
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
