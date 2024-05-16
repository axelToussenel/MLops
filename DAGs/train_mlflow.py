import os
import json
import pandas as pd
import numpy as np
import random
import time
import mlflow
from mlflow import MlflowClient
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from features import FeatureProcessor, FeatureSets

pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160

DATA_PATH = "../dataset"
remote_server_uri = "http://localhost:9090"
experiment_name = "ges_tertiaire"

mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment_name)

mlflow.sklearn.autolog()


def load_data(n_samples, data_dir):
    file_path = os.path.join(data_dir, 'data_pretraite.csv')
    df = pd.read_csv(file_path)

    df = df.sample(n=n_samples)
    df["payload"] = df["payload"].apply(lambda d: json.loads(d))

    data = pd.DataFrame(list(df["payload"].values))
    data.drop(columns="n_dpe", inplace=True)
    data = data.astype(int)
    data = data[data.etiquette_ges > 0].copy()
    data.reset_index(inplace=True, drop=True)
    return data


class NotEnoughSamples(ValueError):
    pass


class TrainGES:
    param_grid = {
        "n_estimators": sorted([random.randint(1, 20)*10 for _ in range(2)]),
        "max_depth": [random.randint(3, 10)],
        "min_samples_leaf": [random.randint(2, 5)],
    }
    n_splits = 3
    test_size = 0.3
    minimum_training_samples = 500

    def __init__(self, data, target="etiquette_dpe"):
        data = data[data[target] >= 0].copy()
        data.reset_index(inplace=True, drop=True)
        if data.shape[0] < TrainGES.minimum_training_samples:
            raise NotEnoughSamples(
                f"data has {data.shape[0]} samples, which is not enough to train a model. min required {TrainGES.minimum_training_samples}"
            )

        self.data = data
        print(f"training on {data.shape[0]} samples")

        self.model = RandomForestClassifier()
        self.target = target
        self.params = {}
        self.train_score = 0.0
        self.precision_score = 0.0
        self.recall_score =  0.0
        self.probabilities =  [0.0, 0.0]

    def main(self):
        X = self.data[FeatureSets.train_columns].copy()
        y = self.data[self.target].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TrainGES.test_size, random_state=808
        )

        cv = KFold(n_splits=TrainGES.n_splits, random_state=42, shuffle=True)

        grid_search = GridSearchCV(
            estimator=self.model, param_grid=TrainGES.param_grid, cv=cv, scoring="accuracy"
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.params = grid_search.best_params_
        self.train_score = grid_search.best_score_
        yhat = grid_search.predict(X_test)
        self.precision_score = precision_score(y_test, yhat, average="weighted")
        self.recall_score = recall_score(y_test, yhat, average="weighted")
        self.probabilities = np.max(grid_search.predict_proba(X_test), axis=1)

    def report(self):
        print("--"*20, "Best model")
        print(f"\tparameters: {self.params}")
        print(f"\tcross-validation score: {self.train_score}")
        print(f"\tmodel: {self.model}")
        print("--"*20, "performance")
        print(f"\tprecision_score: {np.round(self.precision_score, 2)}")
        print(f"\trecall_score: {np.round(self.recall_score, 2)}")
        print(f"\tmedian(probabilities): {np.round(np.median(self.probabilities), 2)}")
        print(f"\tstd(probabilities): {np.round(np.std(self.probabilities), 2)}")

def train_model():
    ctime = int(time.time())
    data = load_data(n_samples=2000, data_dir=DATA_PATH)
    with mlflow.start_run() as run:
        train = TrainGES(data)
        train.main()
        train.report()

# DAG Airflow
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('train_model', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
