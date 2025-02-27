# Databricks notebook source
!pip install /Volumes/mlops_dev/aldrake8/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------

import os

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession

from reservations.config import Config
from reservations.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()

# load config
config = Config.from_yaml(config_path="../project_config.yml", env="dev")

train_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.train_set"
).toPandas()

test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).toPandas()

# COMMAND ----------
data_processor = DataProcessor(input_df=None, config=config)
data_processor.preprocess()

# Generate new data from the original dataset
new_data = data_processor.make_synthetic_data(drift=True, num_rows=200)

# Initialise new DataProcessor
new_processor = DataProcessor(input_df=new_data, config=config)
new_processor.preprocess()

# COMMAND ----------
inference_data_skewed_spark = spark.createDataFrame(new_processor.df) \
    .withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))
    
inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

# COMMAND ----------
