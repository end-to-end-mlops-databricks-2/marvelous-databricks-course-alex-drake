"""
Data loader for retreiving data from Unity Catalog
for use in hotel reservation cancellation project
"""

import pandas as pd
from pyspark.sql import SparkSession
from lightgbm import Dataset

from reservations.config import Config


class DataLoaderUC:
    """
    Custom class for loading data from Unity Catalog
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.schema = f"{config.catalog_name}.{config.schema_name}"
        self.spark = SparkSession.builder.getOrCreate()

        self.train_set = self._load_from_catalog('train_set')
        self.test_set = self._load_from_catalog('test_set')
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

    def _create_lgb_dataset(self, data: pd.DataFrame) -> Dataset:
        """
        Convert pandas.Dataframe to lightgbm.Dataset
        """
        X = data.drop(self.config.target, axis=1)
        y = data[self.config.target]

        return Dataset(X, label=y)

    def prepare_training_data(self):
        """
        Sets the training data as lgb.Dataset

        Creates lgb.Datasets from the training data. To be
        used with lightgbm.train. Not required with
        LGBMClassifier
        """
        self.train_lgb = self._create_lgb_dataset(self.train_set)
        self.test_lgb = self._create_lgb_dataset(self.test_set)
