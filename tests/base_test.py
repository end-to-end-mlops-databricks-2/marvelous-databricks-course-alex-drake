from unittest import TestCase

from src.reservations.config import Config


class BaseTest(TestCase):
    def setUp(self):
        self.config = Config.from_yaml(
            config_path='project_config.yml'
            )
