from unittest import TestCase

from src.reservations.config import Config


class TestConfig(TestCase):
    """Test Config

    Config is an important part of the project, with
    other functions and modules depending on it
    """

    def test_from_yaml(self):
        """
        Test config can be opened from YAML
        """
        config = Config.from_yaml(
            "project_config.yml"
        )

        self.assertEqual(type(config.cat_features), list)
        self.assertEqual(type(config.num_features), list)
