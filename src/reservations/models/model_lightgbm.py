"""
LightGBM Model generation and testing for hotel reservation
cancellation project
"""
import pandas as pd
import numpy as np

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from typing import List

from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

from reservations.config import Config, Tags


class CustomLGBModel:
    """
    A custom class to create a LightGBM model

    :param config: Configuration object for data cleaning.
    """
    def __init__(self, tags: Tags, config: Config) -> None:
        self.config = config
        self.schema = f"{config.catalog_name}.{config.schema_name}"
        self.tags = tags.dict()

        self.experiment_name = self.config.experiment_name

        self.params = config.parameters
        self.model = LGBMClassifier(**self.params)

    def _set_experiment(self):
        """
        Set MLFlow experiment

        Checks if experiment exists, then gets
        or creates
        """
        try:
            mlflow.get_experiment_by_name(self.experiment_name)
        except Exception as e:
            print(f'Error encountered: {e}')
            print('Creating a new experiment')
            mlflow.create_experiment(
                name=self.experiment_name
            )            
        mlflow.set_experiment(
            experiment_name=self.experiment_name
        )

    def train(self, X, y):
        """
        Train function for LGBMClassifier

        :param X: a pd.DataFrame of training data
        :param y: a pd.Series of target data
        """
        self.model.fit(X, y)

    def log_model(self, X, y, code_paths: List[str]):
        """
        Log the model
        """
        self._set_experiment()
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in code_paths:
            whl_name = package.split('/')[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.model.predict(X)
            roc_auc = roc_auc_score(y, y_pred)
            ll = log_loss(y, y_pred)

            metrics = {
                "roc_auc": roc_auc,
                "log_loss": ll
            }

            mlflow.log_param(
                "model_type", "LightGBM"
            )
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(
                model_input=X,
                model_output={'Prediction': 0.0}
            )
            conda_env = _mlflow_conda_env(
                additional_pip_deps=additional_pip_deps
            )

            mlflow.pyfunc.log_model(
                python_model=CustomWrapper(self.model),
                artifact_path="pyfunc-hotel-cancellations-model",
                code_paths=code_paths,
                conda_env=conda_env,
                signature=signature
            )

    def register_model(self):
        """
        Register model in UC
        """
        registered_model = mlflow.register_model(
            model_uri=f'runs:/{self.run_id}/pyfunc-hotel-cancellations-model',
            name=f'{self.schema}.hotel_cancellation_model',
            tags=self.tags
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f'{self.schema}.hotel_cancellation_model',
            alias="latest-model",
            version=latest_version
        )
        
        return latest_version

    def retreive_current_run_metadata(self):
        """
        Get MLflow run metadata
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow with the alias=latest-model and
        make predictions
        """
        model_uri = f"models:/{self.schema}.hotel_cancellation_model@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        predictions = model.predict(input_data)
        return predictions

    def model_improved(self, X, y):
        """
        Evaluate model performance on the test set
        """
        preds_latest = self.load_latest_model_and_predict(X)

        preds_current = self.model.predict(X)

        latest_roc_auc = roc_auc_score(y, preds_latest)
        latest_ll = log_loss(y, preds_latest)
        current_roc_auc = roc_auc_score(y, preds_current)
        current_ll = log_loss(y, preds_current)

        model_status = False
        if current_roc_auc > latest_roc_auc:
            print("Challenger performs better. Register the Challenger.")
            model_status = True
        else:
            print("Champion performs better. Keep the Champion.")

        return model_status


class CustomWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom wrapper for trained model

    Can be used to alter prediction method
    i.e. switch to predict_proba if relevant
    """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame | np.ndarray):
        """
        Run predictions on the input data
        """
        predictions = self.model.predict(model_input)
        return {"Prediction": predictions}
