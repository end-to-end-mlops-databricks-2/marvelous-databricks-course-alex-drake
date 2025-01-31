import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from reservations.config import Config


class DataProcessor:
    """Data Processor

    Pre-process data for use in an ML project.
    Data can be saved to a designated catalog
    within Databricks
    """
    def __init__(self, input_df: pd.DataFrame, config: Config):
        self.config = config

        if input_df:
            self.df = input_df
        else:
            self.get_data_from_config()

        self.train_df = None
        self.test_df = None

    def get_data_from_config(self):
        """Retrieve input data from the config dataset"""
        file_loc = self.config.data
        self.df = pd.read_csv(file_loc)

    def preprocess(self):
        """Preprocess the input DataFrame"""
        target = self.config.target
        self.df[target] = self.df[target].apply(
            lambda x: 0 if x == 'Not_Canceled' else 1
            )

        # treat the numerical features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # treat the categorical features
        cat_features = self.config.cat_features
        for col in cat_features:
            self.df[col] = self.df[col].astype("category")

        # retain the desired features
        id_col = self.config.id_column
        self.df[id_col] = self.df[id_col].astype("str")

        req_columns = [id_col] + num_features + cat_features + [target]
        self.df = self.df[req_columns]

    def split_data(self, test_size=0.3, random_state=42):
        """Split the dataframe into train and test sets"""
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=test_size, random_state=random_state
            )

    def save_df_to_catalog(self, save_df: pd.DataFrame, table_name: str, spark: SparkSession):
        """Save table to Databricks"""

        catalog_destination = (
            f"{self.config.catalog_name}."
            f"{self.config.schema_name}."
            f"{table_name}"
            )

        spark_df = spark.createDataFrame(save_df).withColumn(
            "update_timestamp_utc",
            to_utc_timestamp(current_timestamp(), "UTC")
        )

        spark_df.write.mode("append").saveAsTable(catalog_destination)

        spark.sql(
            f"""
            ALTER TABLE {catalog_destination}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """
        )

    def save_traing_data_to_catalog(self):
        """Save training data to Databricks"""

        if self.train_df:
            self.save_df_to_catalog(
                save_df=self.train_df,
                table_name='train_set',
                spark=SparkSession
                )

        if self.test_df:
            self.save_df_to_catalog(
                save_df=self.test_df,
                table_name='test_set',
                spark=SparkSession
            )
