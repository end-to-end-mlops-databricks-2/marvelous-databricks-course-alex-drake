import argparse
import logging

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from reservations.config import Config
from reservations.serving.model_serving import ModelServing

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(
    taskKey="train_model", key="model_version"
)

# load project config
config = Config.from_yaml(
    config_path=config_path,
    env=args.env
    )
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
model_path = f"{catalog_name}.{schema_name}"
endpoint_name = config.endpoint_name

# Initialise the Model Serving Manager
model_serving = ModelServing(
    model_name=f"{model_path}.hotel_cancellation_model",
    endpoint_name=endpoint_name
)
logger.info("Initialised the Model Serving Manager")

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()
logger.info("Started endpoint deployment...")
