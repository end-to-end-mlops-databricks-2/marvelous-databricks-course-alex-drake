# Databricks notebook source
!pip install /Volumes/mlops_dev/aldrake8/packages/hotel_reservations-latest-py3-none-any.whl

# COMMAND ----------

%restart_python

# COMMAND ----------

import os

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

from reservations.config import Config
from reservations.serving.model_serving import ModelServing
import requests
from typing import List, Dict
import time

# spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get env variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point \
    .getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# load project config
config = Config.from_yaml(config_path="../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
model_path = f"{catalog_name}.{schema_name}"
endpoint_name = config.endpoint_name

# COMMAND ----------
# Initialise the Model Serving manager
model_serving = ModelServing(
    model_name=f"{model_path}.hotel_cancellation_model",
    endpoint_name=endpoint_name
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------
# Create a sample request body
required_cols = config.features
required_cols.remove(config.target)
print(f"{required_cols}")

# COMMAND ----------
# sample test data
test_set = spark.table(f"{model_path}.test_set").toPandas()

sampled_data = test_set[required_cols].sample(
    n=100, replace=True
).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_data]

print(dataframe_records[0])

# COMMAND ----------
# Call the endpoint with a sample record

def call_endpoint(record: List[Dict]):
    """
    Call the model serving endpoint with a given input
    """
    serving_uri = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
    
    response = requests.post(
        serving_uri,
        headers={
            "Authorization": f"Bearer {os.environ['DBR_TOKEN']}"
        },
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# single record response
status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# simplistic load test
for i, data in enumerate(dataframe_records):
    call_endpoint(data)
    time.sleep(0.2)

# COMMAND ----------
# multi-samples call
status_code, response_text = call_endpoint(sampled_data)
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")