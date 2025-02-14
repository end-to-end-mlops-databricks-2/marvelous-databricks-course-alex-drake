from databricks.feature_engineering import (
    FeatureLookup,
    FeatureEngineeringClient
    )
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput
)


class FeatureServing:
    def __init__(self, feature_table_name: str, feature_spec_name: str, 
                 endpoint_name: str):
        """
        Initialise the prediction serving manager
        """
        self.feature_table_name = feature_table_name
        self.workspace = WorkspaceClient()
        self.feature_spec_name = feature_spec_name
        self.online_table = f"{self.feature_table_name}_online"
        self.endpoint_name = endpoint_name
        self.fe = FeatureEngineeringClient()

    def create_online_table(self):
        """
        Creates an online table based on the feature table
        """
        spec = OnlineTableSpec(
            primary_key_columns=["Booking_ID"],
            source_table_full_name=self.online_table,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
                {"triggered":"true"}
                ),
            perform_full_copy=False,
        )
        self.workspace.online_tables.create(name=self.online_table, spec=spec)
        
    def create_feature_spec(self):
        """
        Create feature spec to enable feature serving
        """
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="Booking_ID",
                feature_names=[
                    "arrival_year",
                    "arrival_month"
                ]
            )
        ]
        self.fe.create_feature_spec(name=self.feature_spec_name,
                                    features=features,
                                    exclude_columns=None)
        
    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small",
        scale_to_zero: bool = True):
        """
        Deploys feature serving endpoint in Databricks

        :param workload_size: str. Workload size, or number
        of concurrent requests. Defaults to 4 but can be
        size accordingly (check Databricks documentation)
        :param scale_to_zero: bool. If True, endpoint
        will scale to 0 when not in use.
        """
        endpoint_exists = any(
            item.name == self.endpoint_name for item in self.workspace._serving_endpoints.list()
            )
        
        served_entites = [
            ServedEntityInput(
                entity_name=self.feature_spec_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size
            )
        ]
        
        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entites
                ),
            )
        else:
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=served_entites
            )
