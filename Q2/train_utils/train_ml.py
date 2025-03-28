# import multiple classifiers from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd

import logging

import os
import joblib


FEATURES_FOLDER = 'features'
PARAMS_GRID = {
    'rf': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [10, 20, 30],
        'clf__min_samples_split': [2, 5]
    },
    # 'svm': {
        # 'clf__C': [0.1, 1, 10],
        # 'clf__kernel': ['linear', 'poly', 'rbf'],
        # 'clf__degree': [3, 5, 10]
    # },
    'svm': {
        'clf__C': [0.1],
        'clf__kernel': ['linear'],
        'clf__degree': [3]
    },
    'dt': {
        'clf__max_depth': [10, 20, 30, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__criterion': ['gini', 'entropy']
    },
    'knn': {
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance'],
    }
}


def get_classifier(model_name):
    if model_name == 'rf':
        return RandomForestClassifier()
    elif model_name == 'svm':
        return SVC()
    elif model_name == 'dt':
        return DecisionTreeClassifier()
    elif model_name == 'knn':
        return KNeighborsClassifier()
    else:
        raise ValueError('Invalid model name')


def train_ml_model(data: pd.DataFrame, target: pd.Series, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', get_classifier(model_name))
    ])

    grid_search = GridSearchCV(
        pipeline,
        PARAMS_GRID[model_name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # train the model
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)*100

    y_pred_train = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)*100

    logging.info(f'Hyperparameter Tuning Complete\n')
    logging.info(f'BEST HYPERPARAMETERS:\n{'\n'.join([f'{k}={v}' for k, v in grid_search.best_params_.items()])}\n')
    logging.info(f"Train Accuracy: {train_accuracy}")
    logging.info(f"Test Accuracy: {test_accuracy}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return best_model


def load_data():
    all_data = []
    all_labels = []

    for file_name in os.listdir(FEATURES_FOLDER):
        if file_name.endswith('.csv'):
            label = os.path.splitext(file_name)[0]
            data_csv = pd.read_csv(os.path.join(FEATURES_FOLDER, file_name))
            data_csv.drop(columns=['file'], inplace=True)
            all_data.append(data_csv)
            all_labels.extend([label] * len(data_csv))

    data_df = pd.concat(all_data, ignore_index=True)
    target = pd.Series(all_labels, name='label')

    return data_df, target


def main(model_name):
    logging.info(f'Training {model_name} model\n')

    logging.info('Loading data')
    data, target = load_data()
    logging.info('Data loaded\n')

    logging.info('Tuning hyperparameters on search grid')
    logging.info(f'Hyperparameter Grid:\n{'\n'.join([f'{k}: {v}' for k, v in PARAMS_GRID[model_name].items()])}\n')
    model = train_ml_model(data, target, model_name)


    logging.info('Saving model')
    joblib.dump(model, f'models/{model_name}.pkl')    
    logging.info('Model saved successfully\n')


# if __name__ == '__main__':
#     main()