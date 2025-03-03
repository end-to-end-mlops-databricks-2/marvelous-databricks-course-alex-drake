import argparse
import logging

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from reservations.config import Config, Tags
from reservations.data_loader import DataLoaderUC
from reservations.models.model_lightgbm import CustomLGBModel
from reservations.utils import get_package_version

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

parser.add_argument(
    "--git-sha",
    action="store",
    default=None,
    type=str,
    required=True
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"

config = Config.from_yaml(
    config_path=config_path,
    env=args.env
    )
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {
    "git_sha": args.git_sha,
    "branch": args.branch,
    "job_run_id": args.job_run_id
}
tags = Tags(**tags_dict)

# load data
data_loader = DataLoaderUC(config=config)
X_train = data_loader.train_set.drop(columns=[config.target])
y_train = data_loader.train_set[config.target]

X_test = data_loader.test_set.drop(columns=[config.target])
y_test = data_loader.test_set[config.target]

logger.info("Train shape: %s", X_train.shape)

# initialise model and train
custom_model = CustomLGBModel(
    tags=tags, config=config
)

custom_model.train(X_train, y_train)
logger.info("Model training complete")

package_version = get_package_version()
hotel_package = (
    f'/Volumes/mlops_dev/aldrake8/packages/hotel_reservations-'
    f'{package_version}'
    f'-py3-none-any.whl'
)
code_paths = [hotel_package]

custom_model.log_model(
    X_test, y_test,
    code_paths=code_paths
)

# Evaluate the model
model_improved = custom_model.model_improved(X_test, y_test)

if model_improved:
    latest_version = custom_model.register_model()
    logger.info("New Champion registered. Version: %s", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
