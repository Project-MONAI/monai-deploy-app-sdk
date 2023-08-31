import logging

from app import AIUnetrSegApp

if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    AIUnetrSegApp().run()
    logging.info(f"End {__name__}")
