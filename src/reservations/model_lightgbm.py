import pandas as pd

import mlflow
import mlflow.lightgbm

from mlflow.models import infer_signature
from sklearn.metrics import log_loss, roc_auc_score
from pyspark.sql import SparkSession
from lightgbm import LGBMClassifier, Dataset

from reservations.config import Config

class CustomLGBModel:
    """
    A custom class to create a LightGBM model

    :param config: Configuration object for data cleaning.
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.schema = f"{config.catalog_name}.{config.schema_name}"
        self.spark = SparkSession.builder.getOrCreate()
        self.train_set = self._load_from_catalog('train_set')
        self.test_set = self._load_from_catalog('test_set')
        self.params = config.parameters
        self.experiment_id = self._get_experiment_id()

        self.train_lgb = None
        self.test_lgb = None

    def _load_from_catalog(self, table_name='train_set') -> pd.DataFrame:
        """
        Load data from Databricks Unity Catalog

        :param table_name: name of table to load
        :return: A pandas.DataFrame
        """
        sparkdf = self.spark.table(f"{self.schema}.{table_name}")
        return sparkdf.toPandas()

    def _get_experiment_id(self) -> str:
        """
        Gets the experiment ID for MLFlow
        """
        if mlflow.get_experiment(self.config.experiment_name) is None:
            mlflow.create_experiment(
                name=self.config.experiment_name,
                artifact_location=f"{self.schema}/models"
            )
        experiment_id = mlflow.set_experiment(
            self.config.experiment_name
            ).experiment_id

        return experiment_id

    def create_lgb_dataset(self, data: pd.DataFrame) -> Dataset:
        """
        Convert pandas.Dataframe to lightgbm.Dataset
        """
        X = data.drop(self.config.target, axis=1)
        y = data[self.config.target]

        return Dataset(X, label=y)

    def prepare_training_data(self):
        """
        Sets the training data as lgb.Dataset
        """
        self.train_lgb = self.create_lgb_dataset(self.train_set)
        self.test_lgb = self.create_lgb_dataset(self.test_set)

    def train(self, params=None, nested=False) -> dict:
        """
        Train function for LGBMClassifier

        :param params: a dictionary of parameters. If None, defaults 
        to parameters defined in the config
        :param nested: whether to run MLFlow as a nested run
        """

        X_train = self.train_set.drop(self.config.target, axis=1)
        y_train = self.train_set[self.config.target]
        X_test = self.test_set.drop(self.config.target, axis=1)
        y_test = self.test_set[self.config.target]

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            nested=nested
        ):
            if params is None:
                parameters = self.params
            else:
                parameters = params

            model = LGBMClassifier(parameters)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            ll = log_loss(y_test, y_pred)

            metrics = {
                "val_roc_auc": roc_auc,
                "val_log_loss": ll
            }

            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)

            signature = infer_signature(
                model_input=X_test,
                model_output=y_pred
            )

            mlflow.lightgbm.log_model(
                model,
                artifact_path='lgb-model',
                signature=signature
            )

            return metrics
