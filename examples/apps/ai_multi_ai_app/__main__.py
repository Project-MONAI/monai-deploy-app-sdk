import logging

from app import App

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    App().run()
    logging.info(f"End {__name__}")
