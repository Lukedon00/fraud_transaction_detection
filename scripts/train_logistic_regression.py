# train_logistic_regression.py
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from preprocess import load_and_preprocess_data

# Define the directory where all the .pkl files are located
data_directory = './data/'  # Directory path containing all pkl files
model_directory = './model/' # Directory to save the models

# Load and preprocess data from the directory
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_directory)

# Train KNN model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)

# Evaluate and save results
print("SVM Classification Report:\n", classification_report(y_test, y_pred_logreg))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

# Check if the models directory exists, create it if not
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save model to the models directory
model_path = os.path.join(model_directory, 'logreg_model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(logreg, model_file)

print(f"Model saved to {model_path}")

