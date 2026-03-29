import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def train_model():
    print("--- Étape 1 : Chargement des données ---")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Étape 2 : Configuration de la recherche d'optimisation ---")
    
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(random_state=42))
    ])

    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5],
        'regressor__learning_rate': [0.05, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=base_pipeline, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1,
        verbose=1
    )

    print("--- Étape 3 : Recherche des meilleurs paramètres ---")
    grid_search.fit(X_train, y_train)