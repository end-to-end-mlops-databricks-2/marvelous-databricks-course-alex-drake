"""
Feature Lookup Model generation and testing for hotel reservation
cancellation project
"""
from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from typing import List

from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

from reservations.config import Config, Tags


class FeatureLookUpModel:
    """
    Custom class for Feature Lookup Model
    """
    def __init__(self, config: Config, tags: Tags, spark: SparkSession):
        """
        Initialise with project config
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()
        
        # get relevant settings from config
        self.features = self.config.features
        self.target = self.config.target
        self.params = self.config.parameters
        self.catalog = self.config.catalog_name
        self.schema = self.config.schema_name
        
        # define table names now for later
        self.feature_table = f"{self.catalog}.{self.schema}.customer_features"
        self.function = f"{self.catalog}.{self.schema}.total_bookings"
        
        # mlflow config
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

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

    def create_feature_table(self):
        """
        Create or replace the hotel_features table and populate it.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table}
        (Booking_ID STRING NOT NULL, no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT);
                       """)
        self.spark.sql(f"ALTER TABLE {self.feature_table} ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")
        self.spark.sql(f"ALTER TABLE {self.feature_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table} SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog}.{self.schema}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table} SELECT Booking_ID, no_of_previous_cancellations, no_of_previous_bookings_not_canceled FROM {self.catalog}.{self.schema}.test_set"
        )

    def define_feature_function(self):
        """
        Define a function to calculate number of previous bookings
        """
        self.spark.sql(
            f"""
            CREATE OR REPLACE FUNCTION {self.function}(no_of_previous_cancellations INT, no_of_previous_bookings_not_canceled INT)
            RETURNS INT
            LANGUAGE PYTHON AS
            $$
            return no_of_previous_cancellations + no_of_previous_bookings_not_canceled
            $$
            """
        )

    def load_data(self):
        """
        Load training and testing data from Delta tables
        """
        self.train_set = self.spark.table(
            f"{self.catalog}.{self.schema}.train_set"
        ).drop(
            "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled"
        )
        self.test_set = self.spark.table(
            f"{self.catalog}.{self.schema}.test_set"
        ).toPandas()

        self.train_Set = self.trian_set.withColumn(
            "Booking_ID",
            self.train_set["Booking_ID"].cast("string")
        )

    def feature_engineering(self):
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table,
                    feature_names=["no_of_previous_cancellations","no_of_previous_bookings_not_canceled"],
                    lookup_key="Booking_ID"
                ),
                FeatureFunction(
                    udf_name=self.function,
                    output_name="total_bookings",
                    input_bindings={
                        "no_of_previous_cancellations": "no_of_previous_cancellations",
                        "no_of_previous_bookings_not_canceled": "no_of_previous_bookings_not_canceled"
                    },
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.test_set["total_bookings"] = self.test_set["no_of_previous_cancellations"] + self.test_set["no_of_previous_bookings_not_canceled"]

        self.X_train = self.training_df[self.features +["total_bookings"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.features + ["total_bookings"]]
        self.y_test = self.test_set[self.target]

    def train(self):
        """
        Train the model and log results to MLFlow
        """
        self._set_experiment()
        model = LGBMClassifier(**self.params)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            roc_auc = roc_auc_score(self.y_test, y_pred)
            ll = log_loss(self.y_test, y_pred)

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
                model_input=self.X_test,
                model_output=y_pred
            )

            self.fe.log_model(
                model=model,
                flavor=mlflow.lightgbm,
                artifact_path="lgb-model-fe",
                training_set=self.training_set,
                signature=signature
            )

    def register_model(self):
        """
        Register model in UC
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lgb-model-fe",
            tags=self.tags,
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog}.{self.schema}.hotel_res_model_fe",
            alias="latest-model",
            version=latest_version
        )

    def load_latest_model_and_predict(self, X):
        """
        Load the trained model from MLFlow using Feature Engineering
        Client and make predictions.
        """
        model_uri = f"models:/{self.catalog}.{self.schema}.hotel_res_model_fe@latest-model"
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
