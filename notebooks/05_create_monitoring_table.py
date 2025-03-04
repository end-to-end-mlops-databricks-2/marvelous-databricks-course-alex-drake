# Databricks notebook source
!pip install /Volumes/mlops_dev/aldrake8/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------
import requests
import itertools

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
import numpy as np
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from reservations.config import Config
from reservations.data_processor import DataProcessor

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

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
import time
import datetime
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------
required_columns = config.features
required_columns.remove(config.target)

sampled_skewed_records = new_processor.df[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")

# COMMAND ----------
def call_endpoint(dataframe_record):
    """
    Call the model serving endpoint with a given input
    """
    serving_uri = f"https://{host}/serving-endpoints/ad-hotel-cancellation-serving/invocations"
    
    response = requests.post(
        serving_uri,
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={"dataframe_records": dataframe_record},
    )
    return response

# COMMAND ----------
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    response = call_endpoint(record)
    print(f"Response status: {response.status_code}")
    time.sleep(0.2)

# COMMAND ----------
end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    response = call_endpoint(record)
    print(f"Response status: {response.status_code}")
    time.sleep(0.2)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Refresh Monitoring

# COMMAND ----------
from reservations.monitoring import create_or_refresh_monitoring

create_or_refresh_monitoring(
    config=config, spark=spark, workspace=workspace
)