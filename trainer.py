import joblib
from sklearn.ensemble import RandomForestClassifier  # Changed import to RandomForest
from sklearn.metrics import classification_report
from Backend.datamanagement.datamaker import IMG_SIZE
import csv
import numpy as np
from tqdm import tqdm

#--------------------------
#load data from csv files
#--------------------------

def load_data_from_csv(train_path="Backend/train.csv", test_path="Backend/test.csv"):
    X_train, y_train, X_test, y_test = [], [], [], []
    # Load training data
    counter=0
    with open(train_path, 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        for row in tqdm(reader,desc="Loading training data"):
            X_train.append([float(x) if x != '' else 0 for x in row[:-1] ])  # All but last column
            y_train.append(row[-1])  # Last column
            counter+=1
            if counter==6000:
                break
    # Load testing data
    with open(test_path, 'r',encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header
        for row in tqdm(reader,desc="Loading testing data"):
            X_test.append([float(x) if x != '' else 0 for x in row[:-1]])  # All but last column
            y_test.append(row[-1])  # Last column

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

X_train, y_train, X_test, y_test=load_data_from_csv()
# -------------------------
# Config
# -------------------------
MODEL_PATH = "skin_disease_model.pkl"
# -------------------------
# Train classifier
# -------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)  # Changed to RandomForest
print("Training Random Forest...")
clf.fit(X_train, y_train)

# -------------------------
# Evaluate on test set
# -------------------------
y_pred = clf.predict(X_test)
print("Test Classification Report:")
a=classification_report(y_test, y_pred)
print(a)


# -------------------------
# Save model + config
# -------------------------
joblib.dump((clf, IMG_SIZE), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

with open("report.txt" ,"w") as wi:
    wi.write(a)