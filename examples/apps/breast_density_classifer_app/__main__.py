import logging

from app import BreastClassificationApp

from monai.deploy.logger import load_env_log_level

if __name__ == "__main__":
    load_env_log_level()
    logging.info(f"Begin {__name__}")
    BreastClassificationApp().run()
    logging.info(f"End {__name__}")
