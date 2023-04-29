import logging

from app import AIUnetrSegApp

from monai.deploy.logger import load_env_log_level

if __name__ == "__main__":
    load_env_log_level()
    logging.info(f"Begin {__name__}")
    AIUnetrSegApp().run()
    logging.info(f"End {__name__}")
