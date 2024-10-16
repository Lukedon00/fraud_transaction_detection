# train_knn.py
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from preprocess import load_and_preprocess_data

# Define the directory where all the .pkl files are located
data_directory = './data/'  # Directory path containing all pkl files
model_directory = './model/' # Directory to save the models

# Load and preprocess data from the directory
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_directory)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate and save results
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# Check if the models directory exists, create it if not
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save model to the models directory
model_path = os.path.join(model_directory, 'knn_model.pkl')
with open(model_path, 'wb') as model_file:
    pickle.dump(knn, model_file)

print(f"Model saved to {model_path}")

