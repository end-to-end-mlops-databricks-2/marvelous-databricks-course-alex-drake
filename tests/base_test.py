from unittest import TestCase

from src.reservations.config import Config, Tags


class BaseTest(TestCase):
    def setUp(self):
        self.config = Config.from_yaml(
            config_path='project_config.yml'
            )
        self.tags = Tags(**{
            "git_sha": "abcd1234",
            "branch": "dev"
        })
