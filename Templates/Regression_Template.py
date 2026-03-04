# Import Required Libraries
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Define models for regression
models = {
    "RandomForest": RandomForestRegressor(),
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "SVRRegression": SVR(),
    "DecisionTree": DecisionTreeRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor()
}

# Model Evaluation Function
def evaluate_model(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Training set performance
        train_metrics = {
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'R2': r2_score(y_train, y_train_pred)
        }
        
        # Test set performance
        test_metrics = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R2': r2_score(y_test, y_test_pred)
        }
        
        results[name] = {'train': train_metrics, 'test': test_metrics}
        print(f"Model: {name}\n{'='*60}")
        print("Training Data Output\n", train_metrics)
        print("Testing Data Output\n", test_metrics, "\n")

    return results

# Visualization Function
def plot_results(results):
    model_names = list(results.keys())
    train_r2_scores = [results[name]['train']['R2'] for name in model_names]
    test_r2_scores = [results[name]['test']['R2'] for name in model_names]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=train_r2_scores, color='blue', alpha=0.6, label='Train R2')
    sns.barplot(x=model_names, y=test_r2_scores, color='red', alpha=0.6, label='Test R2')
    plt.xticks(rotation=45)
    plt.title("Model R2 Scores for Train and Test Sets")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.show()

# Hyperparameter Tuning Function
def hyperparameter_tune(models, X_train, y_train):
    with open("./config_regression.json", "r") as file:
        config = json.load(file)

    best_params = {}
    for name, model in models.items():
        if name in config:
            param_grid = config[name]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
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

# Visualize results
plot_results(results)
