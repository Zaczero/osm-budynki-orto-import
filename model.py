import json
from statistics import mean

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import DATA_DIR, SEED
from dataset import load_dataset

_OPTUNA_DB = f'sqlite:///{DATA_DIR}/model_optuna.db'


def _default_params() -> dict:
    return {
        'objective': 'binary',
        'class_weight': 'balanced',
        'force_col_wise': True,
    }


def create_model():
    df = load_dataset()

    X = df.drop(columns='label')
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train.drop(columns='id', inplace=True)

    X_val_id = X_val['id']
    X_val.drop(columns='id', inplace=True)

    def _objective(trial: optuna.Trial) -> float:
        params = _default_params() | {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=5)
        scores = []

        for train_index, val_index in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            lgbm = LGBMClassifier(**params, random_state=SEED, verbose=-1)
            lgbm.fit(X_train_fold, y_train_fold)

            y_pred = lgbm.predict(X_val_fold)
            score = precision_score(y_val_fold, y_pred)
            scores.append(score)

        return mean(scores)

    study = optuna.create_study(
        storage=_OPTUNA_DB,
        pruner=optuna.pruners.MedianPruner(20),
        study_name='',
        load_if_exists=True,
        direction='maximize')

    n_trials = 1000 - len(study.trials)

    if n_trials > 0:
        study.optimize(_objective, n_trials=n_trials, n_jobs=2)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_params)
    print('Best value:', study.best_value)

    with open(DATA_DIR / 'model.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    model = LGBMClassifier(**(_default_params() | study.best_params), random_state=SEED)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    feature_importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print('Top 10 most important features:')
    print(feature_importances_df.head(10))
    print()

    print('Top 10 least important features:')
    print(feature_importances_df.tail(10))
    print()

    y_pred_proba = model.predict_proba(X_val)
    y_pred = y_pred_proba[:, 1] >= 0.8

    val_score = precision_score(y_val, y_pred)
    print(f'Validation score: {val_score:.3f}')
    print()

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    print(f'True Negatives: {tn}')
    print(f'[❗] False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'[✅] True Positives: {tp}')
    print()

    for pred, proba, true, identifier in sorted(zip(y_pred, y_pred_proba, y_val, X_val_id), key=lambda x: x[3].lower()):
        if pred != true and not true:
            print(f'FP: {identifier!r} - {true} != {pred} [{proba[1]:.3f}]')


class Model:
    def __init__(self):
        df = load_dataset()

        X = df.drop(columns=['id', 'label'])
        y = df['label']

        with open(DATA_DIR / 'model.json') as f:
            params = json.load(f)

        self.model = LGBMClassifier(**(_default_params() | params), random_state=SEED)
        self.model.fit(X, y)

    def predict_single(self, X: dict) -> tuple[bool, float]:
        X = pd.DataFrame([X])
        y_pred_proba = self.model.predict_proba(X)
        y_pred = y_pred_proba[:, 1] >= 0.8
        return y_pred[0], y_pred_proba[0, 1]
