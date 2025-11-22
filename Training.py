import pandas as pd
import numpy as np
from NewtonMethod import CustomLogisticRegression
import os

def train_emotion_model(csv_path, model_save_path='models/emotion_model.pth'):
    print(f"Loading data from {csv_path}...")
    
    # 1. Load CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    print(f"Data loaded. Shape: {df.shape}")

    # 2. Separate Features (X) and Labels (y)
    # ASSUMPTION: The label is in a column named 'label' or 'emotion'.
    # If not, we assume it's the LAST column.
    
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label']).values
    elif 'emotion' in df.columns:
        y = df['emotion'].values
        X = df.drop(columns=['emotion']).values
    else:
        print("Warning: 'label' or 'emotion' column not found. Using the last column as label.")
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values

    # Ensure X is float32 (for PyTorch) and y is int (for classes)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    print(f"Features (X): {X.shape}")
    print(f"Labels (y): {y.shape}")
    
    # Check if dimensions match ViT (768)
    if X.shape[1] != 768:
        print(f"WARNING: Input features are size {X.shape[1]}, but ViT usually outputs 768.")
        print("Ensure your CSV contains exactly the extracted features.")

    # 3. Initialize Model
    # input_size must match columns in X, num_classes = 7 (0-6)
    model = CustomLogisticRegression(input_size=X.shape[1], num_classes=7)

    # 4. Train using Newton's Method (L-BFGS) inside .fit()
    print("Starting training...")
    model.fit(X, y, max_iter=100)

    # 5. Save the trained weights
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_weights(model_save_path)
    print("Training complete and model saved!")

if __name__ == "__main__":
    # Replace with your actual CSV filename
    csv_file = "training_features.csv" 
    
    # Create a dummy CSV for testing if it doesn't exist
    if not os.path.exists(csv_file):
        print("Creating dummy CSV for demonstration...")
        data = np.random.randn(100, 768)
        labels = np.random.randint(0, 7, 100).reshape(-1, 1)
        df = pd.DataFrame(np.hstack([data, labels]))
        # Rename last col to 'label'
        df.rename(columns={768: 'label'}, inplace=True)
        df.to_csv(csv_file, index=False)

    train_emotion_model(csv_file)