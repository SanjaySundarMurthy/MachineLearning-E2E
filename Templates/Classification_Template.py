# Import Required Libraries
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize

# Define models for classification
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RidgeClassifier": RidgeClassifier(),
    "SVC": SVC(probability=True),  # SVC with probability=True for ROC AUC
    "DecisionTree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier()
}

# Model Evaluation Function
def evaluate_model(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Train set metrics
        train_metrics = {
            'Accuracy': accuracy_score(y_train, y_train_pred),
            'F1 Score': f1_score(y_train, y_train_pred, average='weighted'),
            'Precision': precision_score(y_train, y_train_pred, average='weighted'),
            'Recall': recall_score(y_train, y_train_pred, average='weighted')
        }

        # Test set metrics
        test_metrics = {
            'Accuracy': accuracy_score(y_test, y_test_pred),
            'F1 Score': f1_score(y_test, y_test_pred, average='weighted'),
            'Precision': precision_score(y_test, y_test_pred, average='weighted'),
            'Recall': recall_score(y_test, y_test_pred, average='weighted')
        }
        
        # ROC AUC Score
        if len(np.unique(y_test)) == 2:  # Binary classification
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_metrics['ROC AUC'] = roc_auc_score(y_test, y_test_proba)
        else:  # Multiclass classification (one-vs-rest approach)
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            y_test_proba = model.predict_proba(X_test)
            test_metrics['ROC AUC'] = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')

        results[name] = {'train': train_metrics, 'test': test_metrics}
        print(f"Model: {name}\n{'='*60}")
        print("Training Data Output\n", train_metrics)
        print("Testing Data Output\n", test_metrics, "\n")
    
    return results

# ROC Curve Plotting Function
def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):  # Ensure model supports predict_proba
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            if y_test_bin.shape[1] == 1:  # Binary classification
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            else:  # Multiclass classification
                y_score = model.predict_proba(X_test)
                fpr = {}
                tpr = {}
                roc_auc = {}
                for i in range(y_test_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], label=f"{name} (Class {i} AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curves for Classification Models")
    plt.legend(loc="lower right")
    plt.show()

# Hyperparameter Tuning Function
def hyperparameter_tune(models, X_train, y_train):
    with open("./config_classifier.json", "r") as file:
        config = json.load(file)

    best_params = {}
    for name, model in models.items():
        if name in config:
            param_grid = config[name]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_params[name] = grid_search.best_params_
            print(f"Model: {name}")
            print(f"Best Parameters: {grid_search.best_params_}\n")
            models[name].set_params(**grid_search.best_params_)
        else:
            print(f"No hyperparameters defined for model: {name}")

    return models

# Sample Usage
# Assuming you have split your data into X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune hyperparameters
models = hyperparameter_tune(models, X_train, y_train)

# Evaluate models
results = evaluate_model(models, X_train, y_train, X_test, y_test)

# Plot ROC AUC curves
plot_roc_curves(models, X_test, y_test)
