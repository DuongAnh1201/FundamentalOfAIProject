import pandas as pd
import numpy as np
from NewtonMethod import CustomLogisticRegression
import os
import matplotlib.pyplot as plt
import torch

# Try to import seaborn, fallback to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Helper functions to replace sklearn
def accuracy_score(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label indices (optional)
    
    Returns:
        confusion_matrix: 2D array
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm


def classification_report(y_true, y_pred, target_names=None):
    """
    Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: List of class names
    
    Returns:
        report_string: Formatted classification report
    """
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    
    if target_names is None:
        target_names = [f'Class {i}' for i in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate metrics for each class
    report_lines = []
    report_lines.append("              precision    recall  f1-score   support\n")
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_support = 0
    
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[i, :].sum()
        
        total_precision += precision * support
        total_recall += recall * support
        total_f1 += f1 * support
        total_support += support
        
        report_lines.append(f"{target_names[i]:15s} {precision:8.2f} {recall:8.2f} {f1:8.2f} {support:8d}\n")
    
    # Calculate weighted averages
    avg_precision = total_precision / total_support if total_support > 0 else 0.0
    avg_recall = total_recall / total_support if total_support > 0 else 0.0
    avg_f1 = total_f1 / total_support if total_support > 0 else 0.0
    
    report_lines.append(f"\n{'accuracy':15s} {'':8s} {'':8s} {accuracy_score(y_true, y_pred):8.2f} {total_support:8d}\n")
    report_lines.append(f"{'macro avg':15s} {avg_precision:8.2f} {avg_recall:8.2f} {avg_f1:8.2f} {total_support:8d}\n")
    report_lines.append(f"{'weighted avg':15s} {avg_precision:8.2f} {avg_recall:8.2f} {avg_f1:8.2f} {total_support:8d}\n")
    
    return ''.join(report_lines)

def load_data(csv_path):
    """
    Load data from CSV file and separate features and labels.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        X: Features array
        y: Labels array
    """
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        return None, None

    print(f"Data loaded. Shape: {df.shape}")

    # Separate Features (X) and Labels (y)
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

    return X, y


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate model on a dataset and return metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        dataset_name: Name of the dataset (for printing)
    
    Returns:
        accuracy: Accuracy score
        predictions: Predicted labels
        probs: Prediction probabilities
    """
    print(f"\n--- Evaluating on {dataset_name} ---")
    
    # Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float()
    else:
        X_tensor = X
    
    if isinstance(y, np.ndarray):
        y_tensor = torch.from_numpy(y).long()
    else:
        y_tensor = y
    
    # Get predictions
    with torch.no_grad():
        logits = torch.matmul(X_tensor, model.W) + model.b
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total samples: {len(y)}")
    
    return accuracy, predictions, probs.cpu().numpy()


def plot_confusion_matrix(y_true, y_pred, emotion_names, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        emotion_names: List of emotion names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_names, yticklabels=emotion_names,
                    cbar_kws={'label': 'Count'})
    else:
        # Use matplotlib's imshow as fallback
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Count')
        tick_marks = np.arange(len(emotion_names))
        plt.xticks(tick_marks, emotion_names, rotation=45)
        plt.yticks(tick_marks, emotion_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(train_losses, val_losses=None, save_path='training_history.png'):
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linewidth=2, markersize=6, 
             label='Training Loss', color='blue')
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, marker='s', linewidth=2, markersize=6, 
                 label='Validation Loss', color='red')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history chart saved to {save_path}")
    plt.close()


def plot_accuracy_comparison(train_acc, val_acc, test_acc, iteration, save_path='accuracy_comparison.png'):
    """
    Plot accuracy comparison across datasets.
    
    Args:
        train_acc: Training accuracy
        val_acc: Validation accuracy
        test_acc: Test accuracy
        save_path: Path to save the figure
    """
    datasets = ['Training', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(datasets, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}\n({acc*100:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Model Accuracy Comparison {iteration}', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy comparison chart saved to {save_path}")
    plt.close()


def train_emotion_model(train_csv_path, iteration, val_csv_path=None, test_csv_path=None, 
                        model_save_path='models/emotion_model.pth', 
                        max_iter=100, epochs=30, output_dir='results'):
    """
    Train emotion detection model with validation and testing.
    
    Args:
        train_csv_path: Path to training CSV file
        val_csv_path: Path to validation CSV file (optional)
        test_csv_path: Path to test CSV file (optional)
        model_save_path: Path to save the trained model
        max_iter: Maximum iterations for L-BFGS
        epochs: Number of training epochs
        output_dir: Directory to save results and charts
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Load training data
    X_train, y_train = load_data(train_csv_path)
    if X_train is None:
        return
    
    # Load validation data if provided
    X_val, y_val = None, None
    if val_csv_path and os.path.exists(val_csv_path):
        X_val, y_val = load_data(val_csv_path)
    
    # Load test data if provided
    X_test, y_test = None, None
    if test_csv_path and os.path.exists(test_csv_path):
        X_test, y_test = load_data(test_csv_path)
    
    # Initialize Model
    model = CustomLogisticRegression(input_size=X_train.shape[1], num_classes=7)
    
    # Train model with validation data
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    train_losses, val_losses = model.fit(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        max_iter=max_iter, epochs=epochs
    )
    
    # Save the trained weights
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_weights(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # Evaluate on training set
    train_acc, train_pred, train_probs = evaluate_model(model, X_train, y_train, "Training Set")
    
    # Evaluate on validation set
    val_acc, val_pred, val_probs = None, None, None
    if X_val is not None and y_val is not None:
        val_acc, val_pred, val_probs = evaluate_model(model, X_val, y_val, "Validation Set")
    
    # Evaluate on test set
    test_acc, test_pred, test_probs = None, None, None
    if X_test is not None and y_test is not None:
        test_acc, test_pred, test_probs = evaluate_model(model, X_test, y_test, "Test Set")
    
    # Emotion names
    emotion_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised']
    
    # Generate charts
    print("\n" + "=" * 70)
    print("Generating Charts...")
    print("=" * 70)
    
    # Plot training history (losses)
    plot_training_history(train_losses, val_losses, 
                         save_path=os.path.join(output_dir, 'training_history.png'))
    
    # Plot accuracy comparison
    if val_acc is not None and test_acc is not None:
        plot_accuracy_comparison(
            train_acc,
            val_acc,
            test_acc,
            iteration,
            save_path=os.path.join(output_dir, 'accuracy_comparison.png')
        )
    elif val_acc is not None:
        # Only train and val
        plot_accuracy_comparison(
            train_acc,
            val_acc,
            train_acc,
            iteration,
            save_path=os.path.join(output_dir, 'accuracy_comparison.png')
        )
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, train_pred, emotion_names, 
                         save_path=os.path.join(output_dir, 'confusion_matrix_train.png'))
    
    if val_pred is not None:
        plot_confusion_matrix(y_val, val_pred, emotion_names, 
                             save_path=os.path.join(output_dir, 'confusion_matrix_val.png'))
    
    if test_pred is not None:
        plot_confusion_matrix(y_test, test_pred, emotion_names, 
                             save_path=os.path.join(output_dir, 'confusion_matrix_test.png'))
    
    # Print classification reports
    print("\n" + "=" * 70)
    print("Classification Report - Training Set")
    print("=" * 70)
    print(classification_report(y_train, train_pred, target_names=emotion_names))
    
    if val_pred is not None:
        print("\n" + "=" * 70)
        print("Classification Report - Validation Set")
        print("=" * 70)
        print(classification_report(y_val, val_pred, target_names=emotion_names))
    
    if test_pred is not None:
        print("\n" + "=" * 70)
        print("Classification Report - Test Set")
        print("=" * 70)
        print(classification_report(y_test, test_pred, target_names=emotion_names))
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    # Example usage with iteration folders
    # You can specify which iteration to use (1-5)
    i = 6
    for iteration in range(1,i):
        train_csv = f"iteration_{iteration}/train_features.csv"
        val_csv = f"iteration_{iteration}/val_features.csv"
        test_csv = f"iteration_{iteration}/test_features.csv"
        
        # Check if iteration folder exists, otherwise use default files
        if not os.path.exists(train_csv):
            print("Iteration folder not found. Using default file names...")
            train_csv = "train_features.csv"
            val_csv = "val_features.csv" if os.path.exists("val_features.csv") else None
            test_csv = "test_features.csv" if os.path.exists("test_features.csv") else None
        
        # Train with validation and testing
        train_emotion_model(
            train_csv_path=train_csv,
            iteration = iteration,
            val_csv_path=val_csv,
            test_csv_path=test_csv,
            model_save_path=f'models/emotion_model_iter{iteration}.pth',
            max_iter=100,
            epochs=75,
            output_dir=f'results_iter{iteration}'
        )