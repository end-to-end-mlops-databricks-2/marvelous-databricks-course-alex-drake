import argparse
import logging
import yaml

from pyspark.sql import SparkSession

from reservations.config import Config
from reservations.data_processor import DataProcessor

logger = logging.getLogger(__name__)

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

config = Config.from_yaml(
    config_path=config_path,
    env=args.env
    )

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Use the data processor to load the original dataset
data_processor = DataProcessor(input_df=None, config=config)
data_processor.preprocess()

# Generate new data from the original dataset
new_data = data_processor.make_synthetic_data(num_rows=100)
logger.info("Synthetic data generated")

# Initialise new DataProcessor
new_processor = DataProcessor(input_df=new_data, config=config)

# Split data
new_processor.split_data(
    test_size=0.3,
    random_state=42
)
logger.info(
    "Train shape: %s", new_processor.train_df.shape
    )
logger.info(
    "Test shape: %s", new_processor.test_df.shape
    )

# Save to catalog
logger.info("Saving data to catalog")
new_processor.save_traing_data_to_catalog()
