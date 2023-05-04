import logging

from app import App

from monai.deploy.logger import load_env_log_level

load_env_log_level()
logging.info(f"Begin {__name__}")
App.run()
logging.info(f"End {__name__}")
