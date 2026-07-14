# train_model.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURATION
# ============================
DATA_DIR = "dataset"            # Parent folder containing 'train' and 'test'
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")   # optional; if missing, use train only
IMG_SIZE = (128, 128)           # resize to reduce compute
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# ============================
# FUNCTION TO LOAD IMAGES & LABELS
# ============================
def load_data_from_dir(data_dir):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for label in class_names:
        class_path = os.path.join(data_dir, label)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels), class_names

# ============================
# LOAD DATA
# ============================
print("Loading training data...")
X_train_raw, y_train_raw, class_names = load_data_from_dir(TRAIN_DIR)
print(f"Found {len(X_train_raw)} training images in classes: {class_names}")

# If test directory exists, load it and merge with train (or we can use it as test set)
if os.path.exists(TEST_DIR):
    print("Loading test data...")
    X_test_raw, y_test_raw, _ = load_data_from_dir(TEST_DIR)
    print(f"Found {len(X_test_raw)} test images.")
    # Combine train and test for a larger dataset (then split again)
    X_all = np.concatenate([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train_raw, y_test_raw])
else:
    X_all, y_all = X_train_raw, y_train_raw

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)

# ============================
# EXTRACT HOG FEATURES
# ============================
def extract_hog_features(images):
    features = []
    for img in images:
        # HOG returns a 1D feature vector
        hog_feat = hog(img, orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       transform_sqrt=True, block_norm='L2-Hys')
        features.append(hog_feat)
    return np.array(features)

print("Extracting HOG features...")
X_features = extract_hog_features(X_all)
print(f"Feature vector size: {X_features.shape[1]}")

# ============================
# TRAIN / TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ============================
# SCALING
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# TRAIN DECISION TREE
# ============================
print("Training Decision Tree classifier...")
clf = DecisionTreeClassifier(max_depth=20, random_state=42, min_samples_split=10)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ============================
# SAVE MODEL & PREPROCESSORS
# ============================
joblib.dump(clf, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model, scaler, and label encoder saved successfully.")