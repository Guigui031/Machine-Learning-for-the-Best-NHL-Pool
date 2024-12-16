from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.svm import SVR
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

import gc
import torch
from GPUtil import showUtilization as gpu_usage


# Définit paramètres de validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error)


def create_pipeline_and_params(model_name):    
    """Définir pipeline et esapce hyperparamétrique"""

    if model_name == 'XGBoost':
        pipeline = Pipeline([
            ('model', XGBRegressor(eval_metric='logloss'))
        ])
        param_grid = {
            'model__learning_rate': uniform(0.01, 0.2),         
            'model__n_estimators': [200, 300, 400, 500],        
            'model__max_depth': [3, 5, 7, 10],                
            'model__subsample': uniform(0.6, 0.4)
        }
    
    elif model_name == 'LogisticRegression':
        pipeline = Pipeline([
            ('model', LogisticRegression(max_iter=1000))
        ])
        param_grid = {'model__C': uniform(0.1, 10), 'model__penalty': ['l1'], 'model__solver': ['liblinear']}

    elif model_name == 'SVR':
        pipeline = Pipeline([
            ('model', SVR(kernel='rbf'))
        ])
        param_grid = {
            'model__C': np.logspace(-3, 3, 10),  # Regularization parameter
            'model__epsilon': np.linspace(0.1, 1.0, 10),  # Epsilon in the loss function
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types
            'model__degree': [2, 3, 4],  # Degree of the polynomial kernel
            'model__gamma': ['scale', 'auto'],  # Kernel coefficient
        }

    elif model_name == 'SGD':
        pipeline = Pipeline([
            ('model', SGDRegressor(loss='modified_huber', penalty='elasticnet', max_iter=10000))
        ])
        param_grid = {
            'model__alpha': np.logspace(-2, 0, 10),
            'model__l1_ratio': np.linspace(0.001, 1, 10)
        }

    else:
        raise ValueError(f"Model {model_name} is not defined.")
    
    return pipeline, param_grid

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    gc.collect()
    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()


def tune_model(pipeline, param_grid, X_train, y_train):
    """Appliquer validation croisée aléatoire à un modèle spécifique"""
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, scoring='neg_mean_squared_error', cv=cv,
        n_iter=10, n_jobs=1, random_state=0, verbose=3
    )
    random_search.fit(X_train, y_train)
    print(f"Best F1 for {pipeline.named_steps['model'].__class__.__name__}:", random_search.best_score_)
    y_pred = cross_val_predict(random_search.best_estimator_, X_train, y_train, cv=cv, method='predict')
    # for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    #     y_val_true = y_train[val_idx]
    #     y_val_pred = y_pred[val_idx]
        # cm = confusion_matrix(y_val_true, y_val_pred)
        # print(f'Confusion Matrix - Fold {fold + 1}:\n{cm}\n')
    best_estimator = random_search.best_estimator_

    return best_estimator

def train_ensemble(X_train, y_train, model_names):
    """Point d'entrée pour entraîner un VotingClassifier"""
    tuned_models = []
    svr = SVR()
    svr.fit(X_train, y_train)
    return svr
    
    for model_name in model_names:
        pipeline, param_grid = create_pipeline_and_params(model_name)
        best_model = tune_model(pipeline, param_grid, X_train, y_train)
        tuned_models.append((model_name.lower(), best_model))

    ensemble = VotingRegressor(estimators=tuned_models)
    ensemble.fit(X_train, y_train)

    return ensemble

def estimate(X_test, model):
    return model.predict(X_test)