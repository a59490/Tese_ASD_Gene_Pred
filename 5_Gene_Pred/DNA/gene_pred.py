import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Load the cat_1_sd dataset
cat_1_sd = pd.read_csv('gene_lists/cat_1_sd.csv.gz', compression='gzip')

# Separate features (X) and target variable (y)
X_train = cat_1_sd.drop(columns=['4'])  # Assuming the target variable is in column '4'
y_train = cat_1_sd['4']

# Apply PCA
pca = PCA(n_components=0.90)
X_train_pca = pca.fit_transform(X_train)

# Initialize and train the logistic regression model
logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_pca, y_train)

# Load the new dataset on which you want to make predictions
new_data = pd.read_csv('path_to_new_dataset.csv')

# Apply the same PCA transformation to the new dataset
X_new_pca = pca.transform(new_data)

# Make probability predictions on the new dataset using the trained logistic regression model
predicted_probabilities = logistic_regression.predict_proba(X_new_pca)

# Optionally, you can add the predicted probabilities to the new dataset and save it
new_data['predicted_probabilities'] = predicted_probabilities[:, 1]  # Probability of positive class
new_data.to_csv('predicted_probabilities.csv', index=False)
