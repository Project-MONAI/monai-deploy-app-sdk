import logging

from app import BreastClassificationApp

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    BreastClassificationApp().run()
    logging.info(f"End {__name__}")
