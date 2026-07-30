import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import contextlib
import csv
from tqdm import tqdm
from itertools import cycle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

# --- Original Code Components ---
MODEL_PATH = "skin_disease_model.pkl"

# Load trained model
# Note: clf will be the scikit-learn classifier, and IMG_SIZE will be the tuple (height, width, channels)
# This assumes the model was saved with joblib.dump((model, IMG_SIZE), MODEL_PATH)
try:
    clf, IMG_SIZE = joblib.load(MODEL_PATH)
    # Check if clf is a scikit-learn classifier with a .classes_ attribute
    if not hasattr(clf, 'classes_'):
        raise TypeError("The loaded model does not appear to be a scikit-learn classifier.")
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --- New Functions for Plotting ---

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generates and saves a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_evaluation.png')
    print("\nConfusion matrix saved as 'confusion_matrix_evaluation.png'")

def plot_roc_auc_curve(y_true, y_score, n_classes, class_names):
    """Generates and saves a multi-class ROC AUC curve plot."""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label='ROC curve of {0} (area = {1:0.2f})'.format(class_names[i], roc_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_auc_curve_evaluation.png')
    print("ROC AUC curve plot saved as 'roc_auc_curve_evaluation.png'")

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Data Loading from CSV ---
    test_path = 'Backend/test.csv'  # Path to your test CSV file
    
    X_test = []
    y_test_str = []


    try:
        with open(test_path, 'r', encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header
            for row in tqdm(reader, desc="Loading testing data"):
                X_test.append([float(x) if x != '' else 0 for x in row[:-1]])
                y_test_str.append(row[-1])

        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{test_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Convert lists to NumPy arrays
    X_test = np.array(X_test)
    
    # Use LabelEncoder to convert string labels to integers
    le = LabelEncoder()
    y_test_int = le.fit_transform(y_test_str)
    class_names = le.classes_
    n_classes = len(class_names)
    print("Class names from data:", class_names)
    
    # --- Model Prediction ---
    print("\nMaking predictions...")
    try:
        with suppress_stdout_stderr():
            # Get predicted class labels for confusion matrix
            y_pred = clf.predict(X_test)  # y_pred is string labels
            
            # Get predicted probabilities for ROC curve
            y_score = clf.predict_proba(X_test)
        print("Predictions completed.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

    # --- Evaluation and Plotting ---
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test_str, y_pred, class_names)  # Use string labels here
    
    print("\nGenerating ROC curve...")
    plot_roc_auc_curve(y_test_int, y_score, n_classes, class_names)  # Use integer labels here