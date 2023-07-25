# german_credit_card_Xgboost
In this code, we first load the German Credit Dataset from a CSV file and preprocess the data by one-hot encoding the categorical features and label encoding the target variable. We then split the dataset into training and testing sets and train an XGBoost classifier on the training set.

Next, we generate predictions on the testing set and evaluate the performance of the XGBoost classifier using confusion matrix and classification report. We then train an Explainable Boosting Classifier (EBC) on the same training set and generate global and local explanations for the model using the explain_global() and explain_local() methods from the interpret library.

Finally, we visualize the global and local explanations using the show() method from the interpret library. The global explanation shows the feature importance and directionality for the entire dataset, while the local explanation shows the feature contributions and decision-making process for a specific loan application.

