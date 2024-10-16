# Fraud Transaction Detection System

## Project Overview
This project aims to classify transactions as fraudulent or legitimate using various machine learning models. The dataset contains transaction details, including customer and terminal information, transaction amount, and timestamps.

Three different models were used for comparison:
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP, Neural Networks)
- Logistic Regression

The models were evaluated using precision, recall, F1-score, and confusion matrices.

## Models
- **KNN**: Simple, easy-to-understand but struggles with imbalanced data.
- **MLP**: Neural network-based, captures complex patterns but is slower to train.
- **Logistic Regression**: Basic and interpretable, with moderate performance.

## Results
(Insert a table of model performance metrics here)

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-transaction-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training scripts:
   ```bash
   python scripts/train_knn.py
   python scripts/train_mlp.py
   python scripts/train_logistic.py
   ```

## Future Work
- Explore ensemble models to improve recall for detecting fraud.
- Use real-world data for a more robust evaluation.
